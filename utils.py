import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_data(data):
    """Reformats nested JSON results into a DataFrame grouped by model."""
    reformatted_data_for_df = {}
    for optimizer_name, models_data in data.items():
        for model_name, data_content in models_data.items():
            if model_name not in reformatted_data_for_df:
                reformatted_data_for_df[model_name] = {}
            # Store optimizer-wise data under each model
            reformatted_data_for_df[model_name][optimizer_name] = data_content
    return pd.DataFrame.from_dict(reformatted_data_for_df, orient='index')


def save_results(results, filename, optim_name, model_name):
    """Save/update accuracy and labeled size results into a JSON file."""

    # Load existing data if available
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Existing JSON file {filename} is corrupted. Creating a new file.")
            data = {}
    else:
        data = {}

    # Create missing optimizer/model entries
    if optim_name not in data:
        data[optim_name] = {model_name: {}}

    if model_name not in data[optim_name]:
        data[optim_name][model_name] = {'labeled_sizes': [], 'accuracies': []}

    # Update values for this optimizer-model combination
    data[optim_name][model_name]['accuracies'] = results[optim_name][model_name]['accuracies']
    data[optim_name][model_name]['labeled_sizes'] = results[optim_name][model_name]['labeled_sizes']

    # Save updated JSON
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results for (Optimizer: {optim_name}, Model: {model_name}) saved to {filename}")


def generate_avg_accuracy(df, models, optimizers):
    """Compute average accuracy across trials for each model–optimizer pair."""
    processed_data = {}
    for model in models:
        processed_data[model] = {}
        for optimizer in optimizers:

            # Skip missing entries
            if optimizer in df.columns and model in df.index:
                data = df.loc[model, optimizer]

                # Get raw accuracy trials and consistent labeled sizes
                accuracies_list = data['accuracies']
                labeled_sizes = data['labeled_sizes'][0]  # assume same for all trials

                # Compute per-step mean accuracy across trials
                accuracies_array = np.array(accuracies_list)
                average_accuracies = np.mean(accuracies_array, axis=0)

                processed_data[model][optimizer] = {
                    'labeled_sizes': labeled_sizes,
                    'average_accuracies': average_accuracies.tolist()
                }
            else:
                print(f"Warning: Missing data for model '{model}' and optimizer '{optimizer}'. Skipping.")

    return processed_data


def plot_all_graph(models, optimizers, processed_data):
    """Plot combined accuracy graph for all models and optimizers."""
    for model in models:
        for optimizer_name in optimizers:
            if optimizer_name in processed_data[model]:
                labeled_sizes = processed_data[model][optimizer_name]['labeled_sizes']
                average_accuracies = processed_data[model][optimizer_name]['average_accuracies']

                # Plot a curve for each model-optimizer pair
                plt.plot(
                    labeled_sizes,
                    average_accuracies,
                    marker='.',
                    label=f'{model.replace("resnet", "ResNet-")} - {optimizer_name.upper()}'
                )

    plt.title('Average Accuracy vs. Labeled Data Size for All Models and Optimizers')
    plt.xlabel('Labeled Data Size')
    plt.ylabel('Average Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save figure
    plot_filename = os.path.join('plots', 'all_comparison.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def plot_accuracy_vs_labeled_size(file_path):
    """Main function: load JSON, compute averages, and generate plots."""
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Load JSON file with experiment results
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return

    # Convert into DataFrame for structured access
    df = format_data(data)

    models = ['resnet18', 'resnet34', 'resnet50']
    optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']

    # Compute average accuracies
    processed_data = generate_avg_accuracy(df, models, optimizers)

    # Plot each model separately
    for model in models:
        plt.figure(figsize=(12, 7))
        for optimizer in optimizers:
            if optimizer in df.columns and model in df.index and optimizer in processed_data[model]:
                plt.plot(
                    processed_data[model][optimizer]['labeled_sizes'],
                    processed_data[model][optimizer]['average_accuracies'],
                    marker='o',
                    label=optimizer.upper()
                )

        plt.title(f'{model.replace("resnet", "ResNet-")} Average Accuracy vs. Labeled Data Size')
        plt.xlabel('Labeled Data Size')
        plt.ylabel('Average Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.ylim(65, 95)
        plt.show()

        # Save model-specific plot
        os.makedirs('plots', exist_ok=True)
        plot_filename = os.path.join('plots', model + '_comparison.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    # Plot combined graph
    plot_all_graph(models, optimizers, processed_data)
