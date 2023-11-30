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