# Python
import os
import random
import json
import matplotlib.pyplot as plt
import argparse

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

# Torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
from dataset import cifar
from dataset.sampler import SubsetSequentialSampler

from tqdm import tqdm

# Custom
import model.resnet as resnet
import model.lossnet as lossnet

# Seed
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)
torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset', type=str, default='cifar10')
    parser.add_argument('--save_path', help='Save Path', type=str, default='results/')

    parser.add_argument('--num_trial', type=int, default=5, help='Number of trials')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the dataloader')
    parser.add_argument('--addednum', type=int, default=1000, help='Budget per cycle')
    parser.add_argument('--cycles', type=int, default=10, help='Total number of cycles')

    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--milestones', type=str, default='160', help='Number of acquisition')

    parser.add_argument('--epoch_loss', type=int, default=120, help='After 120 epochs, stop the gradient from the loss prediction module propagated to the target model')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin')
    parser.add_argument('--weight', type=float, default=1.0, help='Weight')
    parser.add_argument('--subset', type=int, default=10000, help='Subset for learning loss')

    args = parser.parse_args()
    return args


def get_dataset(args):
    if args.dataset in ('cifar10', 'cifar100'):
        dataset = cifar.CIFARDataset(args)
        dataset = dataset.dataset
        args.nTrain, args.nClass = len(dataset['train']), 10
    return dataset, args


def train_epoch(models, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()
    correct, total = 0, 0

    for inputs, labels in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)

        target_loss = criterion(scores, labels)

        _, preds = torch.max(scores.data, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = lossnet.LossPredLoss(pred_loss, target_loss, margin=args.margin)
        loss = m_backbone_loss + args.weight * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

    schedulers['backbone'].step()
    schedulers['module'].step()

    return 100 * correct / total


def test(models, dataloaders, mode='test'):
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    for epoch in range(num_epochs):
        train_acc = train_epoch(models, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss)
    print('>> Finished.')
    return train_acc


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(args.device)
            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def plot_accuracy_graph(file_path):
    """
    Generates and saves a graph comparing accuracy of different active learning methods.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean, professional style
    plt.figure(figsize=(5, 6))

    # Define a color and style for each approach with unique identifiers
    styles = {
        'learning_loss': {'color': 'red', 'linestyle': '-', 'marker': 'o', 'label': 'Learning Loss'},
        # 'learning_loss_95': {'color': 'cyan', 'linestyle': '-', 'marker': 'o', 'label': 'Learning Loss (95%)'},
        # 'learning_loss_80': {'color': 'purple', 'linestyle': '-', 'marker': 'o', 'label': 'Learning Loss (80%)'},
        'dqn': {'color': 'blue', 'linestyle': '-', 'marker': 's', 'label': 'DQN'},
        'userwise_dqn': {'color': 'cyan', 'linestyle': '-', 'marker': 'o', 'label': 'DQN (User-wise)'},
        # 'dqn_dla': {'color': 'green', 'linestyle': '-', 'marker': '^', 'label': 'DQN (DLA)'},
        'non_dqn': {'color': 'orange', 'linestyle': '-', 'marker': 'D', 'label': 'Non-DQN'},
        'userwise_non_dqn': {'color': 'purple', 'linestyle': '-', 'marker': 'o', 'label': 'Non-DQN (User-wise)'},
    }

    # Check if all keys in data exist in the styles dictionary
    for approach in data.keys():
        if approach not in styles:
            print(f"Warning: No style defined for '{approach}'. Skipping this data.")
            continue

        results = data[approach]
        if not results['accuracies']:
            print(f"No data to plot for {approach}.")
            continue

        labeled_sizes = results['labeled_sizes']
        accuracies = results['accuracies']

        # Calculate mean and standard deviation of accuracies across trials
        mean_accuracies = np.round(np.mean(accuracies, axis=0), 2)
        std_accuracies = np.round(np.std(accuracies, axis=0), 2)
        upper_bound = mean_accuracies + std_accuracies
        lower_bound = mean_accuracies - std_accuracies

        style = styles[approach]
        plt.plot(data['learning_loss']['labeled_sizes'][0], mean_accuracies, **style, linewidth=2, markersize=5)
        # plt.fill_between(labeled_sizes, lower_bound, upper_bound, color=style['color'], alpha=0.2)

        # Plot the upper and lower bounds as dotted lines with a single legend entry
        # The first line has a label, and the second has '_nolegend_'
        # plt.plot(data['learning_loss']['labeled_sizes'][0], upper_bound, linestyle=':', color=style['color'], label=f'{style["label"].split(" ")[0]} mean±std')
        # plt.plot(data['learning_loss']['labeled_sizes'][0], lower_bound, linestyle=':', color=style['color'], label='_nolegend_')

    plt.xlabel('Budget', fontsize=12, fontweight='bold')
    plt.ylabel(f'Accuracy', fontsize=12, fontweight='bold')
    plt.title('Active Learning Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Use the first approach's labeled sizes for consistent x-axis ticks
    if 'learning_loss' in data and data['learning_loss']['labeled_sizes'][0]:
        labeled_labels = [f'{size // 1000}K' for size in data['learning_loss']['labeled_sizes'][0]]
        plt.xticks(data['learning_loss']['labeled_sizes'][0], labeled_labels, fontsize=11)

    plt.yticks(fontsize=11)

    # Set the legend location and style
    plt.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=8)
    plt.tight_layout()

    plot_filename = args.save_path + args.dataset + '/active_learning_comparison.png'
    plt.savefig(plot_filename, dpi=300)  # Increase DPI for higher resolution
    print(f"Graph saved as {plot_filename}")
    plt.show()


def save_results(results, filename, optim_name, model_name):
    """
    Saves results to a JSON file, updating accuracies and labeled sizes
    for a specific key, while preserving other keys.

    Args:
        results (dict): The dictionary contains accuracy trials and labeled data sizes.
        filename (str): The path to the JSON file.
        key_to_update (str): The top-level key for the specific active learning approach
                              (e.g., 'learning_loss', 'dqn').
    """
    # Load existing data or initialize an empty dictionary
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Existing JSON file {filename} is corrupted. Creating a new file.")
            data = {}
    else:
        data = {}

    # Initialize the key's dictionary if it doesn't exist
    if optim_name not in data:
        data[optim_name] = {model_name: {}}

    if model_name not in data[optim_name]:
        data[optim_name][model_name] = {'labeled_sizes': [], 'accuracies': []}

    # Update the labeled sizes and accuracies for the specified key
    data[optim_name][model_name]['accuracies'] = results[optim_name][model_name]['accuracies']
    data[optim_name][model_name]['labeled_sizes'] = results[optim_name][model_name]['labeled_sizes']

    # Save the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results for (Optimizer: {optim_name}, Model: {model_name}) saved to {filename}")


if __name__ == '__main__':
    args = get_args()
    args.milestones = list(map(int, args.milestones.split(',')))
    dataset, args = get_dataset(args)
    # Create the base 'results/' directory
    base_save_path = args.save_path
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
        print(f"Created base results directory: {base_save_path}")

    # Create the dataset specific sub-directory (e.g., 'results/cifar10/')
    dataset_save_path = os.path.join(base_save_path, args.dataset)
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
        print(f"Created dataset results directory: {dataset_save_path}")

    # Set the final output filename path
    output_filename = os.path.join(dataset_save_path, 'results.json')

    path = 'weights/'
    model_save_path = os.path.join(path, args.dataset)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created dataset results directory: {model_save_path}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_args = {'batch_size': args.batch_size, 'pin_memory': True}

    rng = random.Random(42)

    ALL_RESULTS = {
       #  'sgd': {
       #       'resnet18': {'label': 'ResNet18', 'labeled_sizes': [], 'accuracies': []},
       #       'resnet34': {'label': 'ResNet34', 'labeled_sizes': [], 'accuracies': []},
       #       'resnet50': {'label': 'ResNet50', 'labeled_sizes': [], 'accuracies': []},
       #  },
        #  'adam': {
         #      'resnet18': {'label': 'ResNet18', 'labeled_sizes': [], 'accuracies': []},
          #    'resnet34': {'label': 'ResNet34', 'labeled_sizes': [], 'accuracies': []},
          #     'resnet50': {'label': 'ResNet50', 'labeled_sizes': [], 'accuracies': []},
       #  },
         # 'adamw': {
         #        'resnet18': {'label': 'ResNet18', 'labeled_sizes': [], 'accuracies': []},
         #        'resnet34': {'label': 'ResNet34', 'labeled_sizes': [], 'accuracies': []},
         #          'resnet50': {'label': 'ResNet50', 'labeled_sizes': [], 'accuracies': []},
         # },
         'rmsprop': {
         #         'resnet18': {'label': 'ResNet18', 'labeled_sizes': [], 'accuracies': []},
          #        'resnet34': {'label': 'ResNet34', 'labeled_sizes': [], 'accuracies': []},
                   'resnet50': {'label': 'ResNet50', 'labeled_sizes': [], 'accuracies': []},
         }
    }

    OPTIMIZER_MAP = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop
    }

    MODEL_MAP = {
        'resnet18': resnet.ResNet18(num_classes=args.nClass).to(args.device),
        'resnet34': resnet.ResNet34(num_classes=args.nClass).to(args.device),
        'resnet50': resnet.ResNet50(num_classes=args.nClass).to(args.device),
    }

    for optim_name, models_data in ALL_RESULTS.items():
        OptimizerClass = OPTIMIZER_MAP[optim_name]  # Get the actual optimizer class

        for model_name, results_data in models_data.items():
            model_path = os.path.join(model_save_path, model_name + '/')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"Created dataset results directory: {model_path}")
            optim_path = os.path.join(model_path, optim_name + '/')
            if not os.path.exists(optim_path):
                os.makedirs(optim_path)
                print(f"Created dataset results directory: {optim_path}")
            for trial in range(args.num_trial):
                print(
                    f"\n--- Running {results_data['label']} (Optimizer: {optim_name}, Model: {model_name}) Approach (Trial {trial + 1}/{args.num_trial}) ---")
                trial_path = os.path.join(optim_path, 't'+str(trial + 1) + '/')
                if not os.path.exists(trial_path):
                    os.makedirs(trial_path)
                    print(f"Created dataset results directory: {trial_path}")

                indices = list(range(args.nTrain))
                rng.shuffle(indices)
                labeled_set = indices[:args.addednum]
                unlabeled_set = indices[args.addednum:]
                trial_accuracies = []
                labeled_sizes = []

                dataloaders = {
                    'train': DataLoader(dataset['train'], sampler=SubsetSequentialSampler(labeled_set), **loader_args),
                    'test': DataLoader(dataset['test'], **loader_args)
                }

                model = MODEL_MAP[model_name]
                loss_module = lossnet.LossNet(model=model_name).to(args.device)
                models = {'backbone': model, 'module': loss_module}
                torch.backends.cudnn.benchmark = False

                for cycle in range(args.cycles):
                    print('Training Set', len(labeled_set))
                    cycle_path = os.path.join(trial_path, 'c' + str(cycle + 1) + '/')
                    if not os.path.exists(cycle_path):
                        os.makedirs(cycle_path)
                        print(f"Created dataset results directory: {cycle_path}")

                    criterion = nn.CrossEntropyLoss(reduction='none')

                    if optim_name == 'sgd':
                        optim_backbone = OptimizerClass(models['backbone'].parameters(), lr=args.lr,
                                                   momentum=args.momentum, weight_decay=args.wdecay)
                        optim_module = OptimizerClass(models['module'].parameters(), lr=args.lr, momentum=args.momentum,
                                                 weight_decay=args.wdecay)
                    elif optim_name == 'adam':
                        optim_backbone = OptimizerClass(models['backbone'].parameters(), lr=0.0001,
                                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
                        optim_module = OptimizerClass(models['module'].parameters(), lr=0.0001,
                                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
                    elif optim_name == 'adamw':
                        optim_backbone = OptimizerClass(models['backbone'].parameters(), lr=0.0001,
                                                        weight_decay=args.wdecay)
                        optim_module = OptimizerClass(models['module'].parameters(), lr=0.0001,
                                                       weight_decay=args.wdecay)
                    elif optim_name == 'rmsprop':
                        optim_backbone = OptimizerClass(models['backbone'].parameters(), lr=0.0001, eps = 1e-08,
                                                        alpha=0.99, momentum=0.9, weight_decay=0.0001)
                        optim_module = OptimizerClass(models['module'].parameters(), lr=0.0001, eps = 1e-08,
                                                      alpha=0.99, momentum=0.9, weight_decay=0.0001)

                    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestones)
                    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=args.milestones)
                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}

                    train(models, criterion, optimizers, schedulers, dataloaders, args.num_epoch, args.epoch_loss)
                    torch.save(model.state_dict(), cycle_path+'model_weights.pth')
                    acc = test(models, dataloaders, mode='test')
                    print(f'Trial {trial + 1}/{args.num_trial} || Cycle {cycle + 1}/{args.cycles} || Labelled Data {len(labeled_set)}: Test Accuracy {acc:.2f}%')

                    labeled_sizes.append(len(labeled_set))
                    trial_accuracies.append(acc)

                    rng.shuffle(unlabeled_set)
                    subset = unlabeled_set[:args.subset]
                    unlabeled_loader = DataLoader(dataset['unlabeled'], sampler=SubsetSequentialSampler(subset), **loader_args)
                    uncertainty = get_uncertainty(models, unlabeled_loader)
                    arg = np.argsort(uncertainty)

                    new_labeled = list(torch.tensor(subset)[arg][-args.addednum:].numpy())
                    labeled_set += new_labeled
                    unlabeled_set = list(torch.tensor(subset)[arg][:-args.addednum].numpy()) + unlabeled_set[args.subset:]

                results_data['labeled_sizes'].append(labeled_sizes)
                results_data['accuracies'].append(trial_accuracies)

                # Save only the learning_loss results after its trials are complete
                save_results(ALL_RESULTS, output_filename, optim_name, model_name)

    # Plot the graph using the saved data
    # plot_accuracy_vs_labeled_size(output_filename)