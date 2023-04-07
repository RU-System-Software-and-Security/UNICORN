from inversion import *
import config
import numpy as np
import sys
import json
import random

from dataloader import get_dataloader_label_partial,get_dataloader_label_remove, get_dataloader_partial_split

import time
    
def main():
    start_time = time.time()
    opt = config.get_argument().parse_args()

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        opt.t_mean = torch.FloatTensor(mean).view(opt.input_channel,1,1).expand(opt.input_channel, opt.input_height, opt.input_width).cuda()
        opt.t_std = torch.FloatTensor(std).view(opt.input_channel,1,1).expand(opt.input_channel, opt.input_height, opt.input_width).cuda()
        
    elif opt.dataset == "imagenet_sub200":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 200
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        opt.t_mean = torch.FloatTensor(mean).view(opt.input_channel,1,1).expand(opt.input_channel, opt.input_height, opt.input_width).cuda()
        opt.t_std = torch.FloatTensor(std).view(opt.input_channel,1,1).expand(opt.input_channel, opt.input_height, opt.input_width).cuda()

    trainset, transform, trainloader, testset, testloader = get_dataloader_partial_split(opt, train_fraction=opt.data_fraction, train=False)
    opt.total_label = opt.num_classes
    opt.test_dataset_total_fixed = trainset
    opt.test_dataloader_total_fixed = trainloader
    data_list = []
    for batch_idx, (inputs, labels) in enumerate(testloader):
        print(batch_idx)
        print(inputs.shape)
        data_list.append(inputs)
    opt.data_test = data_list
    
    data_all_list = []
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        print(batch_idx)
        print(inputs.shape)
        data_all_list.append(inputs)
    opt.data_all = data_all_list
    
    dummy_model = InversionEngine(opt, None, None)
    opt.feature_shape = []
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        features = dummy_model.classifier.from_input_to_features(inputs.cuda(), opt.internal_index)
        for i in range(1, len(features.shape)):
            opt.feature_shape.append(features.shape[i])
        break
    del dummy_model
    init_mask = torch.ones(opt.feature_shape)
    init_pattern = torch.zeros(opt.feature_shape)

    prepare(opt, init_mask, init_pattern)
    
    if opt.set_pair:
        
        print(opt.set_pair.split("_"))
        target = int(opt.set_pair.split("_")[0])
        source = int(opt.set_pair.split("_")[1])
            
        print("----------------- Analyzing pair: target{}, source{}  -----------------".format(target,source))
        opt.target_label = target
        
        test_dataloader = get_dataloader_label_partial(opt,opt.test_dataset_total_fixed,label=source)
        
        testloader_ls = get_dataloader_label_partial(opt,testset,label=source)
        data_list = []
        for batch_idx, (inputs, labels) in enumerate(testloader_ls):
            print(batch_idx)
            print(inputs.shape)
            data_list.append(inputs)
        opt.data_test = data_list
        
        data_list = []
        label_list = []
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            print(batch_idx)
            print(inputs.shape)
            data_list.append(inputs)
            label_list.append(labels)
            
        opt.data_now = data_list
        opt.label_now = label_list
                    
        opt = train(opt, init_mask, init_pattern)
    elif opt.all2one_target:

        target = int(opt.all2one_target)
            
        print("----------------- Analyzing all2one: target{}  -----------------".format(target))
        opt.target_label = target
        
        test_dataloader = get_dataloader_label_remove(opt,opt.test_dataset_total_fixed,label=opt.target_label)
        
        data_list = []
        label_list = []
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            print(batch_idx)
            print(inputs.shape)
            data_list.append(inputs)
            label_list.append(labels)
        opt.data_now = data_list
        opt.label_now = label_list
                    
        opt = train(opt, init_mask, init_pattern)
    


if __name__ == "__main__":
    main()
