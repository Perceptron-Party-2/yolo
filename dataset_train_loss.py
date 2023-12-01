#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multi_digit_dataset import MultiDigitDataset
from yolo_loss import YOLOLoss  # Replace with your YOLO loss class
import wandb
from miniModel import miniModel


# In[2]:


# Configuration parameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 2


# In[ ]:





# In[3]:


import torchvision
import torchvision.transforms as transforms
import random
import torch
from PIL import Image
import numpy
import math



class MultiDigitDataset():
    def __init__(self, grid_size=8, image_width = 800, image_height = 800, 
                 data_size=1000, max_image_number  =  4, num_boxes = 3):
        
        cell_width = 1 / grid_size
        cell_height = 1 / grid_size
        self.canvas_size = (image_width, image_height)
        self.grid_size = grid_size
        self.cell_size = (cell_width, cell_height)
        self.generated_x_values = set()
        self.generated_y_values = set()
        self.data = []
        self.data_size = data_size
        self.num_boxes = num_boxes
        self.max_image_number = max_image_number
        self.generate_canvas_target_pairs()


    def __getitem__(self, index):
        canvas, target = self.data[index]
        return canvas, target

    def __len__(self):
        return self.data_size
   
    def collate_fn(self, batch):
        canvas_pad = torch.nn.utils.rnn.pad_sequence([item['canvas'] for item in batch], batch_first=True, padding_value=0)
        target_pad = torch.nn.utils.rnn.pad_sequence([item['target'] for item in batch], batch_first=True, padding_value=0)
    
        return { 'canvas': canvas_pad, 'target': target_pad }
    
    def yolo_collate_fn(self, batch):
        """
        Custom collate function for YOLO-style datasets.
        Args:
        batch: A list of samples, where each sample is a tuple (image, target).

        Returns:
        A tuple containing a batch of images and targets.
        """
        images = []
        targets = []

        for sample in batch:
            image, target = sample
            images.append(image)
            targets.append(target)

        # Stack images and targets to form batches
        images = torch.stack(images)
        targets = torch.stack(targets)

    

        return images, targets

    def generate_canvas_target_pairs(self):
        
        (image_width, image_height) = self.canvas_size
        
        transform = transforms.Compose([transforms.ToTensor()])

        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        for k in range(0, self.data_size):

            target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes * 15))  # Assuming 5 values (x, y, w, h, class)
            canvas = Image.new('L', self.canvas_size)
            
    
            
            for i in range(0, self.max_image_number):
       
                rand_index = random.randint(0, 100)
        
                # Get MNIST image and label
                img, label = mnist_dataset[rand_index]
       
                 # Convert the tensor image to PIL for processing/
                img = transforms.ToPILImage()(img)
        
                scale_factor = random.uniform(2, 3)
            
                # Randomly resize the MNIST digit
                new_size = tuple([int(scale_factor * s) for s in img.size])
            
                img = img.resize(new_size)
        
                # Choose a random position (where x,y are centred in object) to place the digit

                width, height = img.size
               
                x = random.randint(0, image_width - width)
      
                y = random.randint(0, image_height - height)
        
                  # Place the digit on the canvas
       
                cell_width, cell_height = self.cell_size
               
                
                w_cell = width / image_width
                h_cell = height / image_height
                
               
                
                x_ctr = (x + width/2) 
                
                
                x_ctr /= image_width
                y_ctr = (y + height/2) / image_height
                
              
                  # Determine which grid cell is responsible for the digit
                grid_x = int((x_ctr) // cell_width)
                grid_y = int((y_ctr)// cell_height)
                
                

                      # Create a target tensor representing the grid
                for i in range(0, self.num_boxes):
                    
                    if(target[grid_x, grid_y, i * 15] == 0.0):
                       
                        starting_idx = i * 15
                        canvas.paste(img, (x, y))
        
                        x_ctr_cell = x_ctr - grid_x * cell_width    # Relative x (center) in the responsible cell
       
                        y_ctr_cell = y_ctr  - grid_y * cell_height  # Relative y (center) in the responsible cell
        
                        x_ctr_cell /= cell_width 
            
                        y_ctr_cell /= cell_height 
                       # Define the bounding box (x_center, y_center, width, height) relative to grid cell
                        bounding_box = (x_ctr_cell, y_ctr_cell, w_cell, h_cell)
                        
                        target[grid_x, grid_y, starting_idx] = 1  # Confidence score
                        target[grid_x, grid_y, starting_idx + 1: starting_idx + 5] = torch.tensor(bounding_box)
                        num_classes = 10  
                        
                        one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes)
                        target[grid_x, grid_y, starting_idx + 5:starting_idx + 15] = one_hot_label
                        
                        break
            
                      # Convert the canvas to a tensor
            # Normalize pixel values to be in the range [0, 1]
            
            canvas_tensor = transforms.ToTensor()(canvas)
            canvas_tensor /= 255.0
            self.data.append((canvas_tensor, target))
             
 
    





# In[4]:


wandb.login()


# In[5]:


wandb.init(
    project="yolo",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    },
    group="perceptrongang",

)


# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F

num_anchors = 1

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=10, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.box_loss = 0.0
        self.obj_loss = 0.0
        self.noobj_loss = 0.0
        self.cls_loss = 0.0

    def get_losses(self):
        return self.box_loss, self.obj_loss, self.noobj_loss, self.cls_loss

    def forward(self, output, target):
        step_size = 15 * num_anchors  # Calculate the step size based on the number of anchors
        indices_to_pick = torch.arange(0, 15 * num_anchors, step_size)
        
        # Extract relevant components from the output tensor
        pred_conf, pred_x, pred_y, pred_w, pred_h, pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9 = torch.split(output[:, :, :, indices_to_pick:indices_to_pick + 15], 1, dim=-1)
        #pred_conf = torch.sigmoid(pred_conf)
        
        pred_boxes = torch.cat((pred_x, pred_y, pred_w, pred_h), dim=-1)
        
        #pred_boxes = torch.sigmoid(pred_boxes)
        pred_cls  = torch.cat((pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9), dim=-1)
        #pred_cls = torch.sigmoid(pred_cls)
        # Extract relevant components from the output tensor
        pred_conf = torch.sigmoid(pred_conf)  # Predicted object confidence
        pred_boxes = torch.sigmoid(pred_boxes)  # Predicted bounding boxes (x, y, w, h)
        pred_cls = torch.softmax(pred_cls, dim = -1)  # Predicted class scores

         # ... (rest of the code)

        # Extract relevant components from the target tensor
        true_conf, true_x, true_y, true_w, true_h, true_0, true_1, true_2, true_3, true_4, true_5, true_6, true_7, true_8, true_9 = torch.split(target[:, :, :, indices_to_pick:indices_to_pick + 15], 1, dim=-1)
        true_boxes = torch.cat((true_x, true_y, true_w, true_h), dim=-1)
        true_cls = torch.cat((true_0, true_1, true_2, true_3, true_4, true_5, true_6, true_7, true_8, true_9), dim=-1)

        # Calculate loss components
        box_loss = self.lambda_coord * F.mse_loss(pred_boxes, true_boxes, reduction='mean')
        obj_loss = F.mse_loss(pred_conf, true_conf, reduction='mean')
        noobj_mask = (true_conf == 0.0)
        noobj_loss = self.lambda_noobj * F.mse_loss(pred_conf[noobj_mask], true_conf[noobj_mask], reduction='mean')
        cls_loss = F.cross_entropy(pred_cls, true_cls.argmax(dim=1), reduction='mean')

        # Total loss
        total_loss = box_loss + obj_loss + noobj_loss + cls_loss
        self.box_loss = box_loss
        self.obj_loss = obj_loss
        self.noobj_loss = noobj_loss
        self.cls_loss = cls_loss
        return total_loss


# In[7]:


# Load dataset
grid_size=8
image_width = 800
image_height = 800
data_size=60000
max_image_number  =  1
num_boxes = 2
dataset = MultiDigitDataset(grid_size, image_width, image_height, data_size, max_image_number, num_boxes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.yolo_collate_fn)


# In[8]:


is_mps = torch.backends.mps.is_available()
device = "mps:0" if is_mps else "cpu"
# Initialize model and loss function
model = miniModel(image_width, image_height, image_channels=1, grid_ratio=100, 
                  num_bounding_boxes=1, num_classes=10, dropout=0.5).to(device)   # Fill in with appropriate arguments
criterion = YOLOLoss(num_classes=10).to(device)  # Fill in with appropriate arguments
learning_rate = 0.001
momentum = 0.9
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


# Training loop
for epoch in range(num_epochs):
    i = 1
    for batch in dataloader:
        
        optimizer.zero_grad()
        
        
        images, targets = batch
        
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        # criterion = YOLOLoss(num_classes=10)


        loss = criterion(outputs, targets)
        box_loss, obj_loss, noobj_loss, cls_loss = criterion.get_losses()
        loss.backward()
        optimizer.step()
        # print loss every batch
        print(f"loss: {loss}")
        wandb.log({"loss": loss, "box_loss": box_loss, "cls_loss": cls_loss, " obj_loss": obj_loss})
        
        if (i + 1) % 10 == 0:  # Print every 10th batch
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        i += 1

# Save model
torch.save(model.state_dict(), f"yolo_{epoch}.pth")
print("Training completed.")

wandb.finish()


# In[ ]:





# In[ ]:




