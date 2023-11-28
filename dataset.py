import torchvision
import torchvision.transforms as transforms
import random
from PIL import Image

class MNISTBoundingBoxDataset(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.canvas_size = (700, 200)

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

        # Choose a random position to place the digit
        max_x = self.canvas_size[0] - new_size[0]
        max_y = self.canvas_size[1] - new_size[1]
        top_left_x = random.randint(0, max_x)
        top_left_y = random.randint(0, max_y)

        # Place the digit on the canvas
        canvas.paste(img, (top_left_x, top_left_y))

        # x center
        x = (top_left_x + (top_left_x + new_size[0])) / 2
        # y center
        y = (top_left_y + (top_left_y + new_size[1])) / 2
        # width
        w = new_size[0]
        # height
        h = new_size[1]

        # Define the bounding box (x_min, y_min, x_max, y_max)
        bounding_box = (x, y, w, h)

        # Convert the canvas to a tensor
        canvas = transforms.ToTensor()(canvas)

        return canvas, bounding_box, label

# # Usage
transform = transforms.ToTensor()  # Add any additional transformations here


import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    # Load the dataset
    dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

    # Get a sample image, bounding box, and label
    image, bounding_box, label = dataset[60]  # Change 0 to any index to test different samples

    # Print the label
    print(f"Label: {label}")
    print(f"tensor.shape: {image.shape}")

    # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(pil_img)

    # Add the bounding box
    # Bounding box format: (x_min, y_min, x_max, y_max)
    (x, y, w, h) = bounding_box
    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Display the image and bounding box
    plt.show()
