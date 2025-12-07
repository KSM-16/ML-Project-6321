import torch

def get_uncertainty(args, models, unlabeled_loader):
    """Compute predicted loss (uncertainty) scores for unlabeled data."""

    # Set backbone and prediction module to evaluation mode
    models['backbone'].eval()
    models['module'].eval()

    # Initialize tensor to store uncertainty scores
    uncertainty = torch.tensor([]).to(args.device)

    with torch.no_grad():  # Disable gradients for inference
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(args.device)

            # Forward pass through backbone model
            scores, features = models['backbone'](inputs)

            # Predict loss/uncertainty from extracted features
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))  # Flatten per-sample output

            # Append batch uncertainty values
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()  # Return results on CPU
