# -----------------------------
# Standard Library Imports
# -----------------------------
import os
import random
import argparse

# -----------------------------
# Third-Party Imports
# -----------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# -----------------------------
# Local Project Imports
# -----------------------------
from dataset import cifar
from dataset.sampler import SubsetSequentialSampler
from uncertainty_sampling import get_uncertainty
from model import resnet, lossnet
from train import *
from test import *
from utils import *


# -------------------------------------------------------
# Seeding for reproducibility
# -------------------------------------------------------
def seed_everything(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuner for consistency


# -------------------------------------------------------
# Argument Parser
# -------------------------------------------------------
def get_args():
    """Parse command-line arguments and return configuration."""
    parser = argparse.ArgumentParser()

    # Dataset and saving
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--save_path', type=str, default='results/')

    # Experiment settings
    parser.add_argument('--num_trial', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=200)

    # Active learning settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--addednum', type=int, default=1000)
    parser.add_argument('--cycles', type=int, default=10)

    # Optimization
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--milestones', type=str, default='160')

    # Loss prediction module
    parser.add_argument('--epoch_loss', type=int, default=120)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--subset', type=int, default=10000)

    return parser.parse_args()


# -------------------------------------------------------
# Dataset Loader
# -------------------------------------------------------
def get_dataset(args):
    """Load dataset and attach dataset metadata to args."""
    if args.dataset in ('cifar10', 'cifar100'):
        dataset = cifar.CIFARDataset(args).dataset
        args.nTrain, args.nClass = len(dataset['train']), 10
    return dataset, args


# -------------------------------------------------------
# Main Training/Evaluation Loop
# -------------------------------------------------------
def main(args, dataset, output_filename):
    """Main routine: initialize models, run trials, active learning cycles, and save results."""

    # Directory setup for saving model weights
    model_save_path = os.path.join('weights', args.dataset)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created dataset results directory: {model_save_path}")

    loader_args = {'batch_size': args.batch_size, 'pin_memory': True}
    rng = random.Random(42)

    # Structure for storing results across optimizers & models
    ALL_RESULTS = {
        optim_name: {
            model_name: {'label': model_name.replace("resnet", "ResNet"), 'labeled_sizes': [], 'accuracies': []}
            for model_name in ['resnet18', 'resnet34', 'resnet50']
        }
        for optim_name in ['sgd', 'adam', 'adamw', 'rmsprop']
    }

    # Map optimizer names to PyTorch optimizer classes
    OPTIMIZER_MAP = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop
    }

    # Preload model instances
    MODEL_MAP = {
        name: model.to(args.device)
        for name, model in {
            'resnet18': resnet.ResNet18(num_classes=args.nClass),
            'resnet34': resnet.ResNet34(num_classes=args.nClass),
            'resnet50': resnet.ResNet50(num_classes=args.nClass),
        }.items()
    }

    # ---------------------------------------------
    # Iterate over all optimizers and models
    # ---------------------------------------------
    for optim_name, models_data in ALL_RESULTS.items():
        OptimizerClass = OPTIMIZER_MAP[optim_name]

        for model_name, results_data in models_data.items():

            # Directory setup
            model_path = os.path.join(model_save_path, model_name)
            os.makedirs(model_path, exist_ok=True)

            optim_path = os.path.join(model_path, optim_name)
            os.makedirs(optim_path, exist_ok=True)

            # ---------------------------------------------
            # Multiple trials per configuration
            # ---------------------------------------------
            for trial in range(args.num_trial):
                print(f"\n--- {results_data['label']} | Optimizer: {optim_name} | Trial {trial+1}/{args.num_trial} ---")

                trial_path = os.path.join(optim_path, f"t{trial+1}")
                os.makedirs(trial_path, exist_ok=True)

                # Initial labeled/unlabeled split
                indices = list(range(args.nTrain))
                rng.shuffle(indices)
                labeled_set = indices[:args.addednum]
                unlabeled_set = indices[args.addednum:]

                trial_accuracies = []
                labeled_sizes = []

                # Loaders for train/test sets
                dataloaders = {
                    'train': DataLoader(dataset['train'], sampler=SubsetSequentialSampler(labeled_set), **loader_args),
                    'test': DataLoader(dataset['test'], **loader_args)
                }

                # Initialize backbone and loss prediction model
                model = MODEL_MAP[model_name]
                loss_module = lossnet.LossNet(model=model_name).to(args.device)
                models = {'backbone': model, 'module': loss_module}

                # ---------------------------------------------
                # Active Learning Cycles
                # ---------------------------------------------
                for cycle in range(args.cycles):
                    print(f"Training Set: {len(labeled_set)} samples")

                    cycle_path = os.path.join(trial_path, f"c{cycle+1}")
                    os.makedirs(cycle_path, exist_ok=True)

                    criterion = nn.CrossEntropyLoss(reduction='none')

                    # Select optimizer settings per optimizer type
                    if optim_name == 'sgd':
                        optim_backbone = OptimizerClass(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
                        optim_module = OptimizerClass(loss_module.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
                    elif optim_name == 'adam':
                        optim_backbone = OptimizerClass(model.parameters(), lr=0.0001, weight_decay=0.0001)
                        optim_module = OptimizerClass(loss_module.parameters(), lr=0.0001, weight_decay=0.0001)
                    elif optim_name == 'adamw':
                        optim_backbone = OptimizerClass(model.parameters(), lr=0.0001, weight_decay=args.wdecay)
                        optim_module = OptimizerClass(loss_module.parameters(), lr=0.0001, weight_decay=args.wdecay)
                    elif optim_name == 'rmsprop':
                        optim_backbone = OptimizerClass(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
                        optim_module = OptimizerClass(loss_module.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

                    # Learning rate schedulers
                    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args.milestones)
                    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=args.milestones)

                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}

                    # Train one full epoch set
                    train(args, models, criterion, optimizers, schedulers, dataloaders, args.num_epoch, args.epoch_loss)

                    # Save trained weights
                    torch.save(model.state_dict(), os.path.join(cycle_path, 'model_weights.pth'))

                    # Evaluate performance
                    acc = test(args, models, dataloaders, mode='test')
                    print(f"Cycle {cycle+1}/{args.cycles} | Labeled {len(labeled_set)} | Accuracy: {acc:.2f}%")

                    labeled_sizes.append(len(labeled_set))
                    trial_accuracies.append(acc)

                    # Uncertainty sampling step
                    rng.shuffle(unlabeled_set)
                    subset = unlabeled_set[:args.subset]

                    unlabeled_loader = DataLoader(dataset['unlabeled'],
                                                  sampler=SubsetSequentialSampler(subset),
                                                  **loader_args)

                    uncertainty = get_uncertainty(args, models, unlabeled_loader)
                    arg = np.argsort(uncertainty)

                    # Select most uncertain samples
                    new_labeled = list(torch.tensor(subset)[arg][-args.addednum:].numpy())
                    labeled_set += new_labeled

                    unlabeled_set = list(torch.tensor(subset)[arg][:-args.addednum].numpy()) + unlabeled_set[args.subset:]

                # Save results from this trial
                results_data['labeled_sizes'].append(labeled_sizes)
                results_data['accuracies'].append(trial_accuracies)

                save_results(ALL_RESULTS, output_filename, optim_name, model_name)


# -------------------------------------------------------
# Main Script
# -------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    args.milestones = list(map(int, args.milestones.split(',')))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset, args = get_dataset(args)

    # Create directory structure for results
    os.makedirs(args.save_path, exist_ok=True)
    dataset_save_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(dataset_save_path, exist_ok=True)

    output_filename = os.path.join(dataset_save_path, 'results.json')

    main(args, dataset, output_filename)

    # Plot accuracy curves from stored results
    plot_accuracy_vs_labeled_size(output_filename)
