import torchvision
import torchvision.transforms as transforms
import random
import torch
from PIL import Image

image_width, image_height = 700, 700

class MNISTBoundingBoxDataset(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, grid_size=7):
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
        max_x = self.canvas_size[0] - new_size[0] + new_size[0] / 2
        max_y = self.canvas_size[1] - new_size[1] + new_size[1] / 2
        x = random.randint(0, int(max_x))
        y = random.randint(0, int(max_y))

        # Place the digit on the canvas
        canvas.paste(img, (x, y))

        cell_width, cell_height = self.cell_size
        x_cell = (x / cell_width) % 1  # Relative x (center) in the responsible cell
        y_cell = (y  / cell_height) % 1  # Relative y (center) in the responsible cell
        w_cell = new_size[0] / self.canvas_size[0]  # Relative width to entire image
        h_cell = new_size[1] / self.canvas_size[1]  # Relative height to entire image

        # Determine which grid cell is responsible for the digit
        grid_x = int(x / cell_width)
        grid_y = int(y / cell_height)
        print(f"grid_x: {grid_x}, grid_y: {grid_y}")

        # Define the bounding box (x_center, y_center, width, height) relative to grid cell
        bounding_box = (x_cell, y_cell, w_cell, h_cell)
        print(f"x_cell,y_cell: {x_cell},{y_cell}")

        # Convert the canvas to a tensor
        canvas = transforms.ToTensor()(canvas)

        # Create a target tensor representing the grid
        target = torch.zeros((self.grid_size, self.grid_size, 5))  # Assuming 5 values (x, y, w, h, class)
        target[grid_x, grid_y, :4] = torch.tensor(bounding_box)
        target[grid_x, grid_y, 4] = label  # Add class label

        # target has shape (grid_size, grid_size, 5)
        # print(f"target.shape: {target.shape}")

        return canvas, target

# # Usage
transform = transforms.ToTensor()  # Add any additional transformations here


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to convert relative to absolute bounding boxes
def get_absolute_bounding_box(rel_box, grid_x, grid_y, cell_width, cell_height):
    rel_x_center, rel_y_center, rel_w, rel_h = rel_box
    print(f"rel_x_center, rel_y_center, rel_w, rel_h: {rel_x_center}, {rel_y_center}, {rel_w}, {rel_h}")
    rel_x = rel_x_center - (rel_w / 2)
    rel_y = rel_y_center - (rel_h / 2)
    abs_x = (rel_x + grid_x) * cell_width
    abs_y = (rel_y + grid_y) * cell_height
    abs_w = rel_w * image_width
    abs_h = rel_h * image_height
    x_min = abs_x - (abs_w / 2)
    y_min = abs_y - (abs_h / 2)
    return x_min, y_min, x_min + abs_w, y_min + abs_h


if __name__ == '__main__':
    # Load the dataset
    dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

    # Get a sample image, bounding box, and label
    image, target = dataset[60]  # Change 0 to any index to test different samples

    bounding_box = target[:,:,:4]
    label = target[:,:,4]
    # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(pil_img)

    # Add the grid
    grid_size = 7
    cell_width = image.shape[1] / grid_size
    cell_height = image.shape[0] / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            rel_bounding_box = target[i, j, :4]  # Extract relative bounding box
            abs_bounding_box = get_absolute_bounding_box(rel_bounding_box, i, j, cell_width, cell_height)

            # You can now use abs_bounding_box for plotting or other purposes
            x_min, y_min, x_max, y_max = abs_bounding_box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # Add the bounding box
    # Bounding box format: (x_min, y_min, x_max, y_max)
    # (x, y, w, h) = bounding_box
    # x_min = x - (w / 2)
    # y_min = y - (h / 2)
    # x_max = x + (w / 2)
    # y_max = y + (h / 2)
    # rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
    #                          linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    # Display the image and bounding box
    plt.show()
