import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_class=1.0):
        super(SimpleLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        # Objectness loss
        objectness_loss = self.mse_loss(predictions[..., 0], targets[..., 0])

        # Coordinate loss (x, y, w, h)
        coord_pred = predictions[..., 1:5]
        coord_target = targets[..., 1:5]
        coordinate_loss = self.mse_loss(coord_pred, coord_target)

        # Classification loss (p0, ..., p9)
        class_pred = predictions[..., 5:]
        class_target = targets[..., 5:]
        classification_loss = self.bce_loss(class_pred, class_target)

        # Combining the losses
        total_loss = (self.lambda_coord * coordinate_loss + 
                      self.lambda_obj * objectness_loss + 
                      self.lambda_class * classification_loss)

        # Normalizing the loss by the batch size
        batch_size = predictions.size(0)
        return total_loss / batch_size

# Example usage:
# model_output = ...  # Model predictions, shape [batch_size, 7, 7, 25]
# ground_truth = ...  # Ground truth, shape [batch_size, 7, 7, 25]
# loss_fn = DigitDetectionLoss()
# loss = loss_fn(model_output, ground_truth)
