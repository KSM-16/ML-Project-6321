import torch

def test(args, models, dataloaders, mode='test'):
    """Evaluate backbone model accuracy on validation/test data."""

    # Set both models to evaluation mode
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0

    with torch.no_grad():  # Disable gradients during evaluation
        for (inputs, labels) in dataloaders[mode]:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Forward pass through backbone
            scores, _ = models['backbone'](inputs)

            # Get predicted class
            _, preds = torch.max(scores.data, 1)

            # Update accuracy counters
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total  # Final accuracy percentage
