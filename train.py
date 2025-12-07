import torch
from tqdm import tqdm
from model import lossnet


def train_epoch(args, models, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss):
    """Run one training epoch for backbone + loss prediction module."""

    # Set models to training mode
    models['backbone'].train()
    models['module'].train()

    correct, total = 0, 0  # Track classification accuracy

    # Iterate through training data
    for inputs, labels in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        # Reset gradients
        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # Forward pass on backbone model
        scores, features = models['backbone'](inputs)

        # Compute classification loss
        target_loss = criterion(scores, labels)

        # Compute accuracy
        _, preds = torch.max(scores.data, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Detach features after certain epoch to prevent gradient flow into backbone
        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        # Predict loss using loss prediction module
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        # Loss for backbone + loss prediction module
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = lossnet.LossPredLoss(pred_loss, target_loss, margin=args.margin)
        loss = m_backbone_loss + args.weight * m_module_loss

        # Backpropagation and optimizer update
        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

    # Step learning rate schedulers
    schedulers['backbone'].step()
    schedulers['module'].step()

    return 100 * correct / total  # Return training accuracy


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    """Train for multiple epochs."""
    print('>> Train a Model.')
    for epoch in range(num_epochs):
        train_acc = train_epoch(models, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss)
    print('>> Finished.')
    return train_acc
