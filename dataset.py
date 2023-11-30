import torchvision
import torchvision.transforms as transforms
import random
import torch
from PIL import Image

image_width, image_height = 448, 448
GRID_SIZE = 7 

class MNISTBoundingBoxDataset(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, grid_size=GRID_SIZE):
        super().__init__(root, train=train, transform=transform, download=download)
        cell_width = image_width / grid_size
        self.canvas_size = (image_width, image_height)
        self.grid_size = grid_size
        self.cell_size = (cell_width, cell_width)

    def __getitem__(self, index):
        # Get MNIST image and label
        img, label = super().__getitem__(index)

        # Convert the tensor image to PIL for processing
        img = transforms.ToPILImage()(img)

        scale_factor = random.uniform(1, 4)
        # Randomly resize the MNIST digit
        new_size = tuple([int(scale_factor * s) for s in img.size])
        img = img.resize(new_size)

        # Create a blank canvas
        canvas = Image.new('L', self.canvas_size)

        # Choose a random position (where x,y are centred in object) to place the digit
        half_width, half_height = new_size[0] / 2, new_size[1] / 2
        max_x = self.canvas_size[0] - new_size[0] + half_width
        max_y = self.canvas_size[1] - new_size[1] + half_height
        x_abs = random.randint(int(half_width), int(max_x))
        y_abs = random.randint(int(half_height), int(max_y))

        top_left = (int(x_abs - half_width), int(y_abs - half_height))
        # Place the digit on the canvas
        canvas.paste(img, top_left)

        cell_width, cell_height = self.cell_size
        x = (x_abs / cell_width) % 1  # Relative x (center) in the responsible cell
        y = (y_abs / cell_height) % 1  # Relative y (center) in the responsible cell
        w_cell = new_size[0] / self.canvas_size[0]  # Relative width to entire image
        h_cell = new_size[1] / self.canvas_size[1]  # Relative height to entire image

        # Determine which grid cell is responsible for the digit
        grid_x = int(x_abs / cell_width)
        grid_y = int(y_abs / cell_height)

        # Define the bounding box (x_center, y_center, width, height) relative to grid cell
        bounding_box = (x, y, w_cell, h_cell)

        # Convert the canvas to a tensor
        canvas = transforms.ToTensor()(canvas)

        # Create a target tensor representing the grid
        target = torch.zeros(15, self.grid_size, self.grid_size)  # Assuming 5 values (x, y, w, h, class)
        # Add confidence score (1) to the responsible grid cell
        target[0, grid_x, grid_y] = 1  # Confidence score
        target[1:5, grid_x, grid_y] = torch.tensor(bounding_box)
        num_classes = 10  # Total number of classes (digits 0-9)
        # Convert the label to a one-hot encoded tensor
        one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes)
        # Add the one-hot encoded label to the target tensor
        target[5:, grid_x, grid_y] = one_hot_label


        return canvas, target

# # Usage
transform = transforms.ToTensor()  # Add any additional transformations here

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    # Load the dataset
    dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

    # Get a sample image, bounding box, and label
    image, target = dataset[60]  # Change 0 to any index to test different samples
    bounding_box = target[:4,:,:]
    label = target[4:,:,:]
    # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(pil_img)
    # Add the grid
    cell_width = image_width / GRID_SIZE
    cell_height = image_height / GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rel_bounding_box = target[1:5, i, j]  # Extract relative bounding box
            if(rel_bounding_box[2] > 0 and rel_bounding_box[3] > 0):
                print(f"i: {i}, j: {j}")
                x, y, w, h = rel_bounding_box
                print(f"x: {x}, y: {y}")
                x_abs = (x + i) * cell_width
                y_abs = (y + j) * cell_height
                print(f"cell_width, cell_height: {cell_width}, {cell_height}")
                print(f"x_abs, y_abs: {x_abs}, {y_abs}")
                w_abs = w * image_width
                h_abs = h * image_height
                x_min = x_abs - (w_abs / 2)
                y_min = y_abs - (h_abs / 2)
                print(f"x_min, y_min: {x_min}, {y_min}")
                # You can now use abs_bounding_box for plotting or other purposes
                rect = patches.Rectangle((x_min, y_min), w_abs, h_abs, 
                                        linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    # Display the image and bounding box
    plt.show()
