#!/usr/bin/env python
# coding: utf-8

# In[25]:


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# In[26]:


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool = True):
        super(ConvLayer, self).__init__()
        self.pool = pool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        if self.pool:
            x = self.maxpool(x)
        return x



class miniModel(nn.Module):
    def __init__(self, image_width=448, image_height=448, image_channels=1, grid_ratio=64, num_bounding_boxes=2, num_classes=10, dropout=0.5):
        super(miniModel, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.grid_ratio = grid_ratio
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes
        
        self.layer1 = ConvLayer(in_channels = image_channels, out_channels = 8, kernel_size = 7, stride=1, padding=3, pool=True)
        self.layer2 = ConvLayer(in_channels = 8, out_channels = 16, kernel_size = 3, stride=1, padding=1, pool=True)
        self.layer3 = ConvLayer(in_channels = 16, out_channels = 32, kernel_size = 1, stride=1, padding=0, pool=True)
        self.layer4 = ConvLayer(in_channels = 32, out_channels = 64, kernel_size = 3, stride=1, padding=1, pool=True)
        self.layer5 = ConvLayer(in_channels = 64, out_channels = 128, kernel_size = 3, stride=1, padding=1, pool=True)

        # Fully connected layers
        self.fc1 = nn.Linear(image_width*image_height//8, 1024) # Replace num_cells by right number
        self.fc2 = nn.Linear(1024, (image_width//grid_ratio) * (image_height//grid_ratio) * num_bounding_boxes * (5 + num_classes))  # 1 for confidence, 4 for bounding box coordinates, 10 for classes

        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.view(-1, (self.image_width//self.grid_ratio), (self.image_height//self.grid_ratio), self.num_bounding_boxes * (5 + self.num_classes))  # 7x7 grid, 30 values per grid cell (2 bounding box coordinates + 20 class scores + 2 confidence scores)
        return x


# In[38]:


if __name__ == "__main__":
    model = miniModel()
    print(model)


# In[ ]:




