import torch.nn as nn

class Net(nn.Module):
    def __init__(self,num_classes=10,in_channels=3):
        super(Net, self).__init__()
        use_relu_inplace=False
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.Conv2d(192,  num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=use_relu_inplace),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

            )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        return x
        
    def from_input_to_features(self, x, index):
        x = self.classifier[:19](x)
        #x = x.view(x.size(0), self.num_classes)
        return x
        
    def from_features_to_output(self, x, index):
        x = self.classifier[19:](x)
        x = x.view(x.size(0), self.num_classes)
        return x
        
    def forward_front(self, x):
        x = self.classifier[:19](x)
        #x = x.view(x.size(0), self.num_classes)
        return x
        
    def forward_back(self, x):
        x = self.classifier[19:](x)
        x = x.view(x.size(0), self.num_classes)
        return x
        
    def get_fm(self, x):
        for i in range(23):
            x = self.classifier[i](x)
            if i == 19:
                return x
        
        
    def get_conv_activation(self, x):
        for i in range(18):
            x = self.classifier[i](x)
        return x
        
    def get_all_inner_activation(self, x):
        inner_output_index = [0,2,4,8,10,12,16,18]
        inner_output_list = []
        for i in range(23):
            x = self.classifier[i](x)
            if i in inner_output_index:
                inner_output_list.append(x)
        x = x.view(x.size(0), self.num_classes)
        return x,inner_output_list
        
    def get_all_inner_activation_conv(self, x):
        inner_output_index = [1,3,5,7,9,11,13,15,17,19,22]
        inner_output_list = []
        inner_output_list.append(x)
        for i in range(23):
            x = self.classifier[i](x)
            if i in inner_output_index:
                inner_output_list.append(x)
        x = x.view(x.size(0), self.num_classes)
        return x,inner_output_list
