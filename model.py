import torch
import torch.nn as nn
import torchvision

class YOLOConvNet(nn.Module):
    def __init__(self, conv_configs):
        super(YOLOConvNet, self).__init__()
        
        # Initialize the convolutional layers
        self.conv_layers = nn.ModuleList()
        for conv_config in conv_configs:
            for layer_config in conv_config['conv_layers']:
                conv_layer = nn.Conv2d(in_channels=layer_config['in_channels'],
                                       out_channels=layer_config['out_channels'],
                                       kernel_size=layer_config['kernel_size'],
                                       stride=layer_config['stride'],
                                       padding=layer_config['padding'])
                self.conv_layers.append(conv_layer)
                self.conv_layers.append(nn.LeakyReLU(0.1, inplace=True))

        # Initialize the max pooling layer
        if 'maxpool' in layer_config:
            maxpool_config = layer_config['maxpool']
            maxpool_layer = nn.MaxPool2d(kernel_size=maxpool_config['kernel_size'],
                                            stride=maxpool_config['stride'])
            self.conv_layers.append(maxpool_layer)

    def forward(self, x):
        # Apply the convolutional layers
        # for layer, idx in self.conv_layers:
        for idx, layer in enumerate(self.conv_layers):
            print(f"layer {idx} input shape: {x.shape}")
            x = layer(x)

        return x

class YOLO(torch.nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        # Convolutional layer: Assuming the input image is 28x28x1 (grayscale)
        conv_configs = [{
            'conv_layers': [
                {'in_channels': 1, 'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3},
            ],
            'maxpool': {'kernel_size': 2, 'stride': 2}
        },{
            'conv_layers': [
                {'in_channels': 64, 'out_channels': 192, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'maxpool': {'kernel_size': 2, 'stride': 2}
        },{
            'conv_layers': [
                {'in_channels': 192, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 256, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'maxpool': {'kernel_size': 2, 'stride': 2}
        },{
            'conv_layers': [
                #
                {'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                #
                {'in_channels': 512, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ],
            'maxpool': {'kernel_size': 2, 'stride': 2}
        }, {
            'conv_layers': [
                # Several 1x1 and 3x3 conv layers in repetition
                {'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0},
                {'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                # 
                {'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            ],
        },{
            'conv_layers': [
                # Several 1x1 and 3x3 conv layers in repetition
                {'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
        }]

        self.conv = YOLOConvNet(conv_configs)
        self.fc1 = torch.nn.Linear(in_features=1024 * 7 * 7, out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=7 * 7 * 20)

    def forward(self, x):
        x = self.conv(x)
        print("after: conv", x.shape)
        # Apply adaptive average pooling to reduce each channel to 1x4
        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
        print("after: avgpool", x.shape)
        x = x.view(-1, 1024 * 7 * 7)
        # x = x.view(x.size(0), -1)
        print("after: flattening", x.shape)
        x = self.fc1(x)
        print("after: fc1", x.shape)
        x = self.fc2(x)
        print("after: fc2", x.shape)
        # reshape the output to (batch_size, 7, 7, 20)
        x = x.view(-1, 20, 7, 7)
        print("after: reshape", x.shape)
        return x

