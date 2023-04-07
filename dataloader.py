import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np

from PIL import Image

from io import BytesIO

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

def get_transform(opt, train=True, pretensor_transform=False):
    
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
                
    transforms_list.append(transforms.ToTensor())

    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
        
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_dataloader_partial_split(opt, train_fraction=0.1, train=True, pretensor_transform=False,shuffle=True,return_index = False):
    data_fraction = train_fraction
    
    
    if opt.dataset == "imagenet_sub200":
        class_num=200
        traindir = "/path_to_imagenet_train"
        testdir = "/path_to_imagenet_val"
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        
        transform_train = data_transforms['train']
        transform_test = data_transforms['val']
        
        if train:
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                transform_train
                )
            transform = transform_train
        else:
            dataset = torchvision.datasets.ImageFolder(
                testdir,
                transform_test
                )
            dataset_test = torchvision.datasets.ImageFolder(
                testdir,
                transform_test
                )
            transform = transform_test
    
    else:
        transform_train = get_transform(opt, True, pretensor_transform)
        transform_test = get_transform(opt, False, pretensor_transform)
        
        transform = transform_train
        
        
        if opt.dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform_train, download=True)
            dataset_test = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform_test, download=True)
            class_num=10
        
        else:
            raise Exception("Invalid dataset")

    dataloader_total = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    
    idx = []
    counter = [0]*class_num
    print("dataset.__len__()*data_fraction/class_num:",dataset.__len__()*data_fraction/class_num)
    for batch_idx, (inputs, targets) in enumerate(dataloader_total):
        if counter[targets.item()]<int(dataset.__len__()*data_fraction/class_num):
            idx.append(batch_idx)
            counter[targets.item()] = counter[targets.item()] + 1

    del dataloader_total
    trainset = torch.utils.data.Subset(dataset,idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs,pin_memory=True, num_workers=opt.num_workers, shuffle=shuffle)

    test_idx = list(set(range(dataset.__len__())) - set(idx))
    testset = torch.utils.data.Subset(dataset_test,test_idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs,pin_memory=True, num_workers=opt.num_workers, shuffle=shuffle)

    if return_index:
        return trainset, transform, trainloader, testset, testloader,idx,test_idx
    else:
        return trainset, transform, trainloader, testset, testloader

def get_dataloader_label_partial(opt, dataset_total, label=0):

    dataloader_total = torch.utils.data.DataLoader(dataset_total, batch_size=1,pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    idx = []
    for batch_idx, (inputs, targets) in enumerate(dataloader_total):

        if targets.item() == label:
            idx.append(batch_idx)
    del dataloader_total
    class_dataset = torch.utils.data.Subset(dataset_total,idx)
    dataloader_class = torch.utils.data.DataLoader(class_dataset, batch_size=opt.bs,pin_memory=True,num_workers=opt.num_workers, shuffle=True)

    return dataloader_class

def get_dataloader_label_remove(opt, dataset_total, label=0, idx=None):

    dataloader_total = torch.utils.data.DataLoader(dataset_total, batch_size=1,pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    if idx is None:
        idx = []
        for batch_idx, (inputs, targets) in enumerate(dataloader_total):
            if targets.item() != label:
                idx.append(batch_idx)

    del dataloader_total
    class_dataset = torch.utils.data.Subset(dataset_total,idx)
    dataloader_class = torch.utils.data.DataLoader(class_dataset, batch_size=opt.bs,pin_memory=True,num_workers=opt.num_workers, shuffle=True)

    return dataloader_class

def main():
    pass


if __name__ == "__main__":
    main()
