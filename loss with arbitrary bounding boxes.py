#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
import torch.nn as nn


# In[21]:


def IoU(pred_box, gt_box):
    
    x1 = gt_box[...,0].unsqueeze(dim=-1)
    y1 = gt_box[...,1].unsqueeze(dim=-1)
    w1 = gt_box[...,2].unsqueeze(dim=-1)
    h1 = gt_box[...,3].unsqueeze(dim=-1)
    
    x2 = pred_box[...,0].unsqueeze(dim=-1)
    y2 = pred_box[...,1].unsqueeze(dim=-1)
    w2 = pred_box[...,2].unsqueeze(dim=-1)
    h2 = pred_box[...,3].unsqueeze(dim=-1)
    
    xlg = x1 - 0.5 * w1
    ylg = y1 - 0.5 * h1
    xrg = x1 + 0.5 * w1
    yrg = y1 + 0.5 * h1
    
    xlp = x2 - 0.5 * w2
    ylp = y2 - 0.5 * h2
    xrp = x2 + 0.5 * w2
    yrp = y2 + 0.5 * h2
    
    print(xlg.size())
    print(torch.transpose(xlp, -1,-2).size())
    intersection_x1 = torch.max(xlg, torch.transpose(xlp, -1,-2))
    print(intersection_x1.size())
    intersection_y1 = torch.max(ylg, torch.transpose(ylp, -1,-2))
    
    intersection_x2 = torch.min(xrg, torch.transpose(xrp, -1,-2))
    intersection_y2 = torch.min(yrg, torch.transpose(yrp, -1,-2))

    # Calculate the intersection dimensions
    intersection_width = intersection_x2 - intersection_x1 + 1
    intersection_height = intersection_y2 - intersection_y1 + 1

    # Ensure that intersection dimensions are not less than 0
    intersection_width = torch.clamp(intersection_width, min=0)
    intersection_height = torch.clamp(intersection_height, min=0)

    # Calculate intersection area
    intersection_area = intersection_width * intersection_height
    
    # intersection_area = torch.max(0, intersection_x2 - intersection_x1 + 1) * torch.max(0, intersection_y2 - intersection_y1 + 1)
 
    gt_area = (xrg - xlg + 1) * (yrg - ylg + 1)
    pred_area = (xrp - xlp + 1) * (yrp - ylp + 1)
    
    iou = intersection_area / (gt_area + torch.transpose(pred_area, -1, -2) - intersection_area)

    return iou


# In[23]:


class YOLOLoss(nn.Module):
    def __init__(self, image_width=448, image_height=448, grid_ratio = 64, num_boxes = 2, num_classes=10):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.h_grid = image_width//grid_ratio
        self.v_grid = image_height//grid_ratio
        self.C = num_classes
        self.B = num_boxes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, preds, target):
        preds = preds.reshape(-1, self.h_grid, self.v_grid, self.B, (5 + self.C))
        target = target.reshape(-1, self.h_grid, self.v_grid, self.B, (5 + self.C))
        
        pred_box = pred[...,1:5]
        gt_box = target[...,1:5]

        IoUs = IoU(pred_box, gt_box) # (batch_size, h_grid, v_grid, num_boxes, num_boxes) with the IoUs of each predicted and target box
        max_values, _ = torch.max(IoUs, dim=-1) # (batch_size, h_grid, v_grid, num_boxes) with the indices of the best predicted boxes per target box 
        best_boxes = (IoUs == max_values) # (batch_size, h_grid, v_grid, num_boxes, num_boxes) with the indices of the best boxes being represented as 1s in those indices of the final dimension

        exists_box = target[..., 0].unsqueeze(-1) # (batch_size, h_grid, v_grid, num_boxes, 1)
        
        predicted_coords = preds[..., 1:5]

        target_coords = target[..., 1:5]
        
        box_predictions = exists_box * np.matmul(best_boxes, predicted_coords)

        box_targets = exists_box * np.matmul(best_boxes, target_coords)
        
        box_predictions[..., 3:5] = torch.sign(box_predictions[...,3:5]) * torch.sqrt(torch.abs(box_predictions[...,3:5] + 1e-6))
        
        box_targets[..., 3:5] = torch.sqrt(box_targets[..., 3:5])
        
        box_loss = self.mse(torch.flatten(box_predictions), torch.flatten(box_targets))
        
        predicted_confidences = preds[...,0].unsqueeze(-1)

        target_confidences = target[...,0].unsqueeze(-1)
        
        object_loss = self.mse(torch.flatten(exists_box*torch.matmul(best_boxes, predicted_confidences), torch.flatten(exists_box*torch.matmul(best_boxes, target_confidences))))
        
        no_object_loss = self.mse(torch.flatten((1 - exists_box)*predicted_confidences), torch.flatten((1-exists_box)*target_confidences))

        predicted_classes = preds[..., 5:]

        target_classes = target[..., 5:]
                      
        class_loss = self.mse(torch.flatten(exists_box*predicted_classes), torch.flatten(exists_box*target_classes))
        
        loss = (self.lambda_coord*box_loss + object_loss + self.lambda_noobj*no_object_loss + class_loss)
        
        return loss


# In[ ]:




