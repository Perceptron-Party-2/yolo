#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
import torchvision.transforms as transforms
import random
import torch
from PIL import Image
import numpy
import math



class MultiDigitDataset():
    def __init__(self, grid_size=8, image_width = 800, image_height = 800, data_size=1000, max_image_number  =  4):
        
        cell_width = image_width / grid_size
        self.canvas_size = (image_width, image_height)
        self.grid_size = grid_size
        self.cell_size = (cell_width, cell_width)
        self.generated_x_values = set()
        self.generated_y_values = set()
        self.data_list = []
        self.data_size = data_size
        self.generate_canvas_target_pairs()
        

    def reset(self):
        
        self.generated_x_values = set()
        self.generated_y_values = set()

    def __getitem__(self, index):
        canvas, target = self.data_list[index]
        return { 'canvas': canvas, 'target': target }

    def __len__(self):
        return self.data_size
   
    def collate_fn(self, batch):
        canvas_pad = torch.nn.utils.rnn.pad_sequence([item['canvas'] for item in batch], batch_first=True, padding_value=0)
        target_pad = torch.nn.utils.rnn.pad_sequence([item['target'] for item in batch], batch_first=True, padding_value=0)
    
        return { 'canvas': canvas_pad, 'target': target_pad }

    def generate_canvas_target_pairs(self):
        transform = transforms.Compose([transforms.ToTensor()])

        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        for k in range(0, self.data_size):

            target = torch.zeros((self.grid_size, self.grid_size, 5))  # Assuming 5 values (x, y, w, h, class)
            canvas = Image.new('L', self.canvas_size)
            
            for i in range(0, max_image_number):
       
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

                #Make sure the number images do not overlap. When generating random positions add interval between generated random numbers.
                #The grid cell control is not done in this part. The following does not ensure that only one image will be placed in a grid cell.
            
                while(True):
             
                   rand_x = random.randint(0, 4)
             
                   if rand_x not in self.generated_x_values:
                      self.generated_x_values.add(rand_x)
                      break
                   else: continue
        
                while(True):
             
                    rand_y = random.randint(0, 4)
                    if rand_y not in self.generated_y_values:
                       self.generated_y_values.add(rand_y)
                       break
                    else: continue   

        
                x = rand_x * 100 + width + random.randint(0, 10)
      
                y = rand_y * 100 + height + random.randint(0, 10)
        
                  # Place the digit on the canvas
       
       
                cell_width, cell_height = self.cell_size
                w_cell = width 
                h_cell = height
                x_ctr = (x + width/2)
                y_ctr = (y + width/2)
        
                  # Determine which grid cell is responsible for the digit
                grid_x = int((x_ctr ) / cell_width)
                grid_y = int((y_ctr  )/ cell_height)
        
                  # Make sure two digit images are not placed in the same grid cell. 
                if target[grid_x, grid_y, 2] != 0.0 and target[grid_x, grid_y, 3] != 0.0:
                  
                    continue
                else:
                      canvas.paste(img, (x, y))
        
                      x_ctr_cell = x_ctr - grid_x * cell_width    # Relative x (center) in the responsible cell
       
                      y_ctr_cell = y_ctr  - grid_y * cell_height  # Relative y (center) in the responsible cell
        
       
                       # Define the bounding box (x_center, y_center, width, height) relative to grid cell
                      bounding_box = (x_ctr_cell, y_ctr_cell, w_cell, h_cell)
          
                      # Convert the canvas to a tensor
                      canvas_tensor = transforms.ToTensor()(canvas)

                      # Create a target tensor representing the grid
        
                      target[grid_x, grid_y, :4] = torch.tensor(bounding_box)
                      target[grid_x, grid_y, 4] = label  # Add class label

            self.data_list.append((canvas_tensor, target))
             
            self.reset()
    





# In[2]:


grid_size=8
image_width = 800
image_height = 800
data_size=1000
max_image_number  =  4
dataset = MNISTBoundingBoxDataset(grid_size=8, image_width = 800, image_height = 800, data_size=1000, max_image_number  =  4)
my_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


# In[3]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to convert relative to absolute bounding boxes
def get_absolute_bounding_box(rel_box, grid_x, grid_y, cell_width, cell_height):
    rel_x_center, rel_y_center, rel_w, rel_h = rel_box
    abs_x_center = rel_x_center + cell_width * grid_x
    abs_y_center = rel_y_center + cell_height * grid_y
    x_min = abs_x_center - (rel_w / 2)
    y_min = abs_y_center - (rel_h / 2)
    x_max = abs_x_center + (rel_w / 2)
    y_max = abs_y_center + (rel_h / 2)
    return x_min, y_min, rel_w, rel_h


for batch in my_dataloader:
    
    image = batch['canvas'][0]
    print(image.shape)
    target = batch['target'][0]
     # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")
    
    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(pil_img)

    # Add the grid
   
    width, height = pil_img.size
    cell_width = width / grid_size
    cell_height = height / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            rel_bounding_box = target[i, j, :4]  # Extract relative bounding box
            abs_bounding_box = get_absolute_bounding_box(rel_bounding_box, i, j, cell_width, cell_height)

            # You can now use abs_bounding_box for plotting or other purposes
            x_min, y_min, x_max, y_max = abs_bounding_box
           
            rect = patches.Rectangle((x_min.tolist(), y_min.tolist()), x_max.tolist() , y_max.tolist() , 
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    
    plt.show()


# In[ ]:




