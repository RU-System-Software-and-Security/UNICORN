import torch
from torch import Tensor, nn
import torchvision
import torch.functional as F
import torchvision.transforms as transforms
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from piqa import SSIM
import torch.nn.functional as F

import unet_model
import copy
import random
import pilgram
from PIL import Image
from functools import reduce

from models import nin,vgg,resnet,wresnet,inception,densenet,mobilenetv2,efficientnet


def back_to_np_4d(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]

    elif opt.dataset in ["imagenet_sub200"]:
        expected_values = [0.4802, 0.4481, 0.3975]
        variance = [0.2302, 0.2265, 0.2262]
    inputs_clone = inputs.clone()

    for channel in range(3):
        inputs_clone[:,channel,:,:] = inputs_clone[:,channel,:,:] * variance[channel] + expected_values[channel]

    return inputs_clone*255
    
def np_4d_to_tensor(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset in ["imagenet_sub200"]:
        expected_values = [0.4802, 0.4481, 0.3975]
        variance = [0.2302, 0.2265, 0.2262]
    inputs_clone = inputs.clone().div(255.0)


    for channel in range(3):
        inputs_clone[:,channel,:,:] = (inputs_clone[:,channel,:,:] - expected_values[channel]).div(variance[channel])
    return inputs_clone

def plant_sin_trigger(img, delta=20, f=6, debug=False):

    alpha = 0.7
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j, k] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    return img

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class InversionEngine(nn.Module):
    def __init__(self, opt, init_mask, init_pattern):
        self._EPSILON = opt.EPSILON
        super(InversionEngine, self).__init__()
        
        if init_mask is not None:
            self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
            self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))
            
        
        init_mask_input = torch.ones((opt.input_channel,opt.input_height,opt.input_width))
        init_pattern_input = torch.zeros((opt.input_channel,opt.input_height,opt.input_width))
        
        self.mask_tanh_input = nn.Parameter(torch.tensor(init_mask_input))
        self.pattern_tanh_input = nn.Parameter(torch.tensor(init_pattern_input))

        self.classifier = self._get_classifier(opt)
        
        self.example_features = None

        self.encoder = unet_model.UNet(n_channels=3,num_classes=3,base_filter_num=opt.ae_filter_num, num_blocks=opt.ae_num_blocks)
        self.decoder = unet_model.UNet(n_channels=3,num_classes=3,base_filter_num=opt.ae_filter_num, num_blocks=opt.ae_num_blocks)
        
        self.encoder.train()
        self.decoder.train()
        self.example_ori_img = None
        self.example_ae_img = None
        
        self.example_space_before_img = None
        self.example_space_after_img = None
        self.example_space_trigger = None
        self.opt = opt

        if "wanet" in opt.model_path:
            ckpt_path = opt.model_path
            state_dict = torch.load(ckpt_path)
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
            grid_temps = (identity_grid + 0.5 * noise_grid / opt.input_height) * 1
            grid_temps = torch.clamp(grid_temps, -1, 1)
            opt.grid_temps = grid_temps
        
    def forward_ori(self, x,opt):
    
        features = self.classifier.from_input_to_features(x, opt.internal_index)
        out = self.classifier.from_features_to_output(features, opt.internal_index)
        
        return out, features
        
    def forward_with_trigger(self, x,opt):
    
        self.example_ori_img = x
        x_before_ae = x
        
        mask_input = self.get_raw_mask_input()
        pattern_input = self.get_raw_pattern_input()
        x = self.encoder(x)
        self.example_space_before_img = x
        
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min)/(x_max - x_min)
        
        x = (1 - mask_input) * x + mask_input * pattern_input
        x = x*(x_max - x_min) + x_min
        
        
        self.example_space_after_img = x
        self.example_space_trigger = mask_input * pattern_input

        x = self.decoder(x)
        
        x_after_ae = x
        self.example_ae_img = x
    
        features = self.classifier.from_input_to_features(x, opt.internal_index)
        out = self.classifier.from_features_to_output(features, opt.internal_index)           
        self.example_features = features
        
        return out, features, x_before_ae, x_after_ae


    def forward_with_trigger_with_disen(self, x,opt):
        mask = self.get_raw_mask(opt)

        self.example_ori_img = x
        x_before_ae = x
        
        mask_input = self.get_raw_mask_input()
        pattern_input = self.get_raw_pattern_input()
        x = self.encoder(x)
        self.example_space_before_img = x
        
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min)/(x_max - x_min)
        
        x = (1 - mask_input) * x + mask_input * pattern_input
        x = x*(x_max - x_min) + x_min
        
        
        self.example_space_after_img = x
        self.example_space_trigger = mask_input * pattern_input
        x = self.decoder(x)
        
        x_after_ae = x
        self.example_ae_img = x
            
        features = self.classifier.from_input_to_features(x, opt.internal_index)
        ori_features = self.classifier.from_input_to_features(x_before_ae, opt.internal_index)
        bs = features.shape[0]
        index_1 = list(range(bs))
        random.shuffle(index_1)
        reference_features = ori_features[index_1]
        features_ori = features
        features = mask * features + (1-mask) * reference_features.reshape(features.shape)
        out = self.classifier.from_features_to_output(features, opt.internal_index)  
        
        self.example_features = features_ori
        
        return out, features, x_before_ae, x_after_ae, features_ori
        
    def get_raw_mask_input(self):
        mask = nn.Tanh()(self.mask_tanh_input)
        return mask / (2 + self._EPSILON) + 0.5
        
    def get_raw_pattern_input(self):
        pattern = nn.Tanh()(self.pattern_tanh_input)
        return pattern / (2 + self._EPSILON) + 0.5
        
    def get_raw_mask(self,opt):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        
        return bounded

    def _get_classifier(self, opt):
        if opt.arch:

            if opt.arch == "wresnet-34-1":
                classifier = wresnet.WideResNet(depth=34, num_classes=opt.num_classes, widen_factor=1, dropRate=0.0).to(opt.device)
            
            if opt.arch == "vgg16":
                classifier = vgg.vgg16(num_classes=opt.num_classes).to(opt.device)
                
            if opt.arch == "densenet121":
                classifier = densenet.DenseNet121(num_classes=opt.num_classes).to(opt.device)
                
            if opt.arch == "mobilenetv2":
                classifier = mobilenetv2.MobileNetV2(num_classes=opt.num_classes).to(opt.device)
                
            if opt.arch == "efficientnet":
                classifier = efficientnet.EfficientNetB0(num_classes=opt.num_classes).to(opt.device)
                
            if opt.arch == "inception_v3":
                classifier = inception.inception_v3().to(opt.device)
                
            if opt.arch == "resnet18":
                classifier = resnet.resnet18(num_classes=opt.num_classes).to(opt.device)
                
            if opt.arch == "nin":
                classifier = nin.Net(num_classes=opt.num_classes).to(opt.device)

            if opt.arch == "resnet18_torchvision":
                classifier = torchvision.models.resnet18(True)
                num_class = 200
                classifier.fc = nn.Linear(classifier.fc.in_features, num_class)
                classifier = classifier.to(opt.device)
                print(classifier)
            
            if opt.arch == "vgg16_torchvision":
                classifier = torchvision.models.vgg16(True)
                num_class = 200
                classifier.classifier[6] = nn.Linear(classifier.classifier[6].in_features, num_class)
                classifier = classifier.to(opt.device)

            if opt.arch == "densenet161_torchvision":
                classifier = torchvision.models.densenet161(True)
                num_class = 200
                classifier.classifier = nn.Linear(classifier.classifier.in_features, num_class)
                classifier = classifier.to(opt.device)

        if opt.model_path:
            ckpt_path = opt.model_path

            
            state_dict = torch.load(ckpt_path)
              
            try:
                classifier.load_state_dict(state_dict["state_dict"])
            except:
                            
                try:
                    classifier.load_state_dict(state_dict["net_state_dict"])
                except:
                    try:
                        classifier.load_state_dict(state_dict["netC"])
                    except:
                        try:
                            from collections import OrderedDict
                            new_state_dict = OrderedDict()
                            for k, v in state_dict["model"].items():
                                name = k[7:] # remove `module.`
                                new_state_dict[name] = v
                            classifier.load_state_dict(new_state_dict)
                            
                        except:
                            classifier.load_state_dict(state_dict)
                
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(opt.device)

def train(opt, init_mask, init_pattern):

    data_now = opt.data_now
    
    opt.weight_p = 1
    opt.weight_acc = 1
    opt.weight_std = 1
    
    opt.init_mask = init_mask
    opt.init_pattern = init_pattern
            
    inversion_engine = InversionEngine(opt, init_mask, init_pattern).to(opt.device)

    opt.lr = 1e-3
    
    optimizer_encoder_decoder = torch.optim.Adam([{'params': inversion_engine.encoder.parameters()},{'params': inversion_engine.decoder.parameters()}],lr=opt.lr,betas=(0.5,0.9))
    optimizer_hp_input = torch.optim.Adam([inversion_engine.mask_tanh_input,inversion_engine.pattern_tanh_input],lr=1e-1,betas=(0.5,0.9))    

    optimizerR_mask = torch.optim.Adam([inversion_engine.mask_tanh],lr=1e-1,betas=(0.5,0.9))    

    inversion_engine.encoder.train()
    inversion_engine.decoder.train()
        
    process = inversion
        
    for i in range(500):
        for inputs in data_now:
            inputs = inputs.cuda()
            inversion_engine.encoder.train()
            inversion_engine.decoder.train()
            inversion_engine.mask_tanh_input.requires_grad = False
            inversion_engine.pattern_tanh_input.requires_grad = False
            inversion_engine.mask_tanh.requires_grad = False
            
            optimizer_encoder_decoder.zero_grad()
            x = inversion_engine.encoder(inputs)
            
            x_min = x.min()
            x_max = x.max()
            x = (x - x_min)/(x_max - x_min)
            x = x*(x_max - x_min) + x_min
            
            identity_mapping_inputs = inversion_engine.decoder(x)
            mapping_loss = torch.nn.MSELoss(size_average = True).cuda()(identity_mapping_inputs,inputs)
            total_loss = mapping_loss
            print("mapping_loss:",mapping_loss)

            total_loss.backward()
            optimizer_encoder_decoder.step()
    
    for epoch in range(opt.epoch):
        early_stop = process(inversion_engine, optimizer_encoder_decoder, optimizer_hp_input,optimizerR_mask, data_now, epoch, opt)

        if (epoch+1)%1==0:
            torchvision.utils.save_image(inversion_engine.example_ae_img, "./encoder_decoder_re_imgs/ae_img_label"+str(opt.target_label)+"_ploss"+str(opt.ssim_loss_bound)+"_masksize"+str(opt.mask_size)+"_stdloss"+str(opt.loss_std_bound)+".jpg", normalize=True)
            torchvision.utils.save_image(inversion_engine.example_ori_img, "./encoder_decoder_re_imgs/ori_img_label"+str(opt.target_label)+".jpg", normalize=True)
            print("inversion_engine.example_space_trigger.max():",inversion_engine.example_space_trigger.max())
        if early_stop:
            break

    opt.trained_inversion_engine = inversion_engine

    return opt

def prepare(opt, init_mask, init_pattern):

    test_dataset = opt.test_dataset_total_fixed
    test_dataloader = opt.test_dataloader_total_fixed
    
    inversion_engine = InversionEngine(opt, init_mask, init_pattern).to(opt.device)

    features_list = []
    features_list_class = [[] for i in range(opt.num_classes)]
    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(opt.device)
        out, features = inversion_engine.forward_ori(inputs,opt)
        print(torch.argmax(out,dim=1))

        features_list.append(features)
        for i in range(inputs.shape[0]):
            features_list_class[labels[i].item()].append(features[i].unsqueeze(0))
    all_features = torch.cat(features_list,dim=0)
    print(all_features.shape)

    del features_list
    del test_dataloader

    feature_mean_class_list = []
    for i in range(opt.num_classes):
        feature_mean_class = torch.cat(features_list_class[i],dim=0).mean(0)
        feature_mean_class_list.append(feature_mean_class)

    opt.feature_mean_class_list = feature_mean_class_list
    del all_features
    del features_list_class
    
def inversion(inversion_engine, optimizer_encoder_decoder, optimizer_hp_input,optimizerR_mask, data_now, epoch, opt, warm_up=False):
    print("Epoch {} - Label: {} | {}:".format(epoch, opt.target_label, opt.dataset))
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0
    
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []
    
    loss_reg_ori_list = []
    ssim_loss_list = []
    loss_mask_norm_list = []
    loss_std_list = []
        
    for inputs in data_now:
        
        inversion_engine.encoder.train()
        inversion_engine.decoder.train()
        inversion_engine.mask_tanh_input.requires_grad = False
        inversion_engine.pattern_tanh_input.requires_grad = False
        inversion_engine.mask_tanh.requires_grad = False
        
        optimizer_encoder_decoder.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
        
        predictions, features, x_before_ae, x_after_ae, features_ori = inversion_engine.forward_with_trigger_with_disen(inputs,opt)
        loss_ce = cross_entropy(predictions, target_labels)

        t_mean = opt.t_mean
        t_std = opt.t_std
        
        ssim_loss = SSIMLoss().cuda()(torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1),torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1))
        loss_reg_ori = torch.norm(inversion_engine.get_raw_mask(opt), opt.use_norm)
        dist_loss = torch.cosine_similarity(opt.feature_mean_class_list[opt.target_label].reshape(-1),features_ori.mean(0).reshape(-1),dim=0)

        loss_acc_list_ = []
        minibatch_accuracy_ = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
        loss_acc_list_.append(minibatch_accuracy_)
        loss_acc_list_ = torch.stack(loss_acc_list_)
        avg_loss_acc_G = torch.mean(loss_acc_list_)
        
        loss_acc_list.append(minibatch_accuracy_)
                
        ssim_loss_bound = opt.ssim_loss_bound
        loss_std_bound = opt.loss_std_bound
        
        atk_succ_threshold = 0.9
    
        loss_std = (features_ori*inversion_engine.get_raw_mask(opt)).std(0).sum()
        loss_std = loss_std/(torch.norm(inversion_engine.get_raw_mask(opt), 1))
        
        loss_reg = dist_loss
        total_loss = dist_loss*opt.dist_loss_weight
        if dist_loss<0:
            total_loss = total_loss - dist_loss*opt.dist_loss_weight
        if loss_std>loss_std_bound:
            total_loss = total_loss + loss_std*10*(1+opt.weight_std)
        if (ssim_loss>ssim_loss_bound):
            total_loss = total_loss + ssim_loss*10*(1+opt.weight_p)
        if avg_loss_acc_G.item()<atk_succ_threshold:
            total_loss = total_loss + 1*loss_ce*(1+opt.weight_acc)

        x = inversion_engine.encoder(inputs)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min)/(x_max - x_min)
        x = x*(x_max - x_min) + x_min
        identity_mapping_inputs = inversion_engine.decoder(x)
        out_identity, _ = inversion_engine.forward_ori(identity_mapping_inputs,opt)        
        
        mapping_loss = torch.nn.MSELoss(size_average = True).cuda()(identity_mapping_inputs,inputs)
        total_loss = total_loss + mapping_loss*200
        print("mapping_loss:",mapping_loss)

        total_loss.backward()
        optimizer_encoder_decoder.step()

        #*******************************************************************
        
        inversion_engine.encoder.eval()
        inversion_engine.decoder.eval()
        inversion_engine.mask_tanh_input.requires_grad = True
        inversion_engine.pattern_tanh_input.requires_grad = True
        
        inversion_engine.mask_tanh.requires_grad = False
        
        for k in range(10):
            optimizer_hp_input.zero_grad()

            inputs = inputs.to(opt.device)
            sample_num = inputs.shape[0]
            target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
            
            predictions, features, x_before_ae, x_after_ae, features_ori = inversion_engine.forward_with_trigger_with_disen(inputs,opt)

            loss_ce = cross_entropy(predictions, target_labels)

            t_mean = opt.t_mean
            t_std = opt.t_std
            
            ssim_loss = SSIMLoss().cuda()(torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1),torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1))
            
            loss_reg_ori = torch.norm(inversion_engine.get_raw_mask(opt), opt.use_norm)

            dist_loss = torch.cosine_similarity(opt.feature_mean_class_list[opt.target_label].reshape(-1),features_ori.mean(0).reshape(-1),dim=0)

            loss_acc_list_ = []
            minibatch_accuracy_ = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
            loss_acc_list_.append(minibatch_accuracy_)
            loss_acc_list_ = torch.stack(loss_acc_list_)
            avg_loss_acc_G = torch.mean(loss_acc_list_)
            
            loss_acc_list.append(minibatch_accuracy_)
                                    
            ssim_loss_bound = opt.ssim_loss_bound
            loss_std_bound = opt.loss_std_bound
            atk_succ_threshold = 0.9
            
            loss_std = (features_ori*inversion_engine.get_raw_mask(opt)).std(0).sum()
            loss_std = loss_std/(torch.norm(inversion_engine.get_raw_mask(opt), 1))
            
            loss_reg = dist_loss
            total_loss = dist_loss*opt.dist_loss_weight
            if dist_loss<0.5:
                total_loss = total_loss - dist_loss*opt.dist_loss_weight
            if loss_std>loss_std_bound:
                total_loss = total_loss + loss_std*10*(1+opt.weight_std)
            if (ssim_loss>ssim_loss_bound):
                total_loss = total_loss + ssim_loss*10*(1+opt.weight_p)
                
            if avg_loss_acc_G.item()<atk_succ_threshold:
                total_loss = total_loss + 1*loss_ce*(1+opt.weight_acc)
                    
            space_mask_norm = torch.norm(inversion_engine.get_raw_mask_input(), opt.use_norm)

            space_mask_norm_bound = opt.input_channel*opt.input_height*opt.input_width*0.15
            if space_mask_norm>space_mask_norm_bound:
                total_loss = total_loss + 10*space_mask_norm

            total_loss.backward()
            optimizer_hp_input.step()
        
        #*******************************************************************
        mask_norm_bound = int(reduce(lambda x,y:x*y,opt.feature_shape)*opt.mask_size)
        
        for k in range(10):
            inversion_engine.encoder.eval()
            inversion_engine.decoder.eval()
            inversion_engine.mask_tanh_input.requires_grad = False
            inversion_engine.pattern_tanh_input.requires_grad = False
            inversion_engine.mask_tanh.requires_grad = True
            
            optimizerR_mask.zero_grad()
            predictions, features, x_before_ae, x_after_ae, features_ori = inversion_engine.forward_with_trigger_with_disen(inputs,opt)

            loss_mask_ce = cross_entropy(predictions, target_labels)
            loss_mask_norm = torch.norm(inversion_engine.get_raw_mask(opt), opt.use_norm)

            loss_mask_total = loss_mask_ce
            if loss_mask_norm>mask_norm_bound:
                loss_mask_total = loss_mask_total + loss_mask_norm
                                
            loss_mask_total.backward()
            optimizerR_mask.step()

        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        
        loss_reg_ori_list.append(loss_reg_ori.detach())

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
        total_pred += sample_num
        
        ssim_loss_list.append(ssim_loss)
        loss_mask_norm_list.append(loss_mask_norm)
        loss_std_list.append(loss_std)

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)
    loss_reg_ori_list = torch.stack(loss_reg_ori_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)
    
    avg_loss_reg_ori = torch.mean(loss_reg_ori_list)
    
    opt.use_valacc = False
    if opt.dataset == "imagenet_sub200":
        test_freq = 1
    else:
        test_freq = 5
    
    if (epoch+1)%test_freq == 0:
        with torch.no_grad():
            print("train minibatch_accuracy:",minibatch_accuracy)
            print("train dist_loss:",dist_loss)

            inversion_engine.encoder.eval()
            inversion_engine.decoder.eval()
            inversion_engine.mask_tanh_input.requires_grad = False
            inversion_engine.pattern_tanh_input.requires_grad = False
            inversion_engine.mask_tanh.requires_grad = False
            
            total_pred = 0
            true_pred = 0
        
            ssim_total = 0
            L2_loss_total = 0
            sim_feature_GT2RE_total = 0

            if opt.dataset == "imagenet_sub200":
                test_batch_nums = 10
            else:
                test_batch_nums = 2
            for inputs in opt.data_test[:test_batch_nums]:

                inputs = inputs.to(opt.device)
                sample_num = inputs.shape[0]
                total_pred += sample_num
                target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
                predictions, features, x_before_ae, x_after_ae = inversion_engine.forward_with_trigger(inputs,opt)
                true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
                loss_ce = cross_entropy(predictions, target_labels)

                ssim = 1-SSIMLoss().cuda()(torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1),torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1))
                print("ssim:",ssim)
                L2_loss_before_preprocess = (torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1) - torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1)).pow(2).sum(-1).sum(-1).sum(-1).sqrt()
                L2_loss = L2_loss_before_preprocess.mean()

                ssim_total = ssim_total + ssim.detach().cpu().item()
                L2_loss_total = L2_loss_total + L2_loss.detach().cpu().item()
                
                #inputs
                if "wanet" in opt.model_path:
                    GT_img = F.grid_sample(inputs, opt.grid_temps.repeat(inputs.shape[0], 1, 1, 1), align_corners=True)
                elif "bpp" in opt.model_path:
                    squeeze_num = 32
                    inputs_bd = back_to_np_4d(inputs,opt)
                    inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255
                    GT_img = np_4d_to_tensor(inputs_bd,opt)
                else:
                    GT_img = inputs
                    GT_img = (torch.clamp(GT_img*t_std+t_mean, min=0, max=1).detach().cpu().numpy()*255).astype(np.uint8)
                    for j in range(GT_img.shape[0]):
                        ori_pil_img = Image.fromarray(GT_img[j].transpose((1,2,0)))
                        if "filter" in opt.model_path:
                            convered_pil_img = pilgram._1977(ori_pil_img)
                        elif "patch" in opt.model_path:
                            if opt.dataset == "imagenet_sub200":
                                trigger_size=24
                                position=15
                            else:
                                trigger_size=3
                                position=2
                            up_img = Image.new("RGB", (trigger_size,trigger_size), 'yellow')
                            ori_pil_img.paste(up_img, (position, position))
                            convered_pil_img = ori_pil_img
                        elif "blend" in opt.model_path:
                            width = opt.input_width
                            height = opt.input_height
                            trigger = Image.open("./blend_trigger/hello_kitty.jpg").resize((width, height))
                            ori_pil_img = ori_pil_img.convert('RGB')
                            trigger = trigger.convert('RGB')
                            ori_pil_img = Image.blend(ori_pil_img, trigger, 0.2)
                            convered_pil_img = ori_pil_img
                        elif "sig" in opt.model_path:
                            if opt.dataset == "imagenet_sub200":
                                ori_pil_img = plant_sin_trigger(ori_pil_img, delta=40, f=6, debug=False)
                            else:
                                ori_pil_img = plant_sin_trigger(ori_pil_img, delta=20, f=6, debug=False)
                            ori_pil_img = Image.fromarray(ori_pil_img)
                            convered_pil_img = ori_pil_img
                        GT_img[j] = np.asarray(convered_pil_img).transpose((2,0,1))
                    GT_img = GT_img.astype(np.float32)
                    GT_img = GT_img/255
                    GT_img = torch.from_numpy(GT_img).cuda()
                    GT_img = (GT_img-t_mean)/t_std
                
                out_GT, feature_GT = inversion_engine.forward_ori(GT_img,opt)

                sim_feature_GT2benign = torch.cosine_similarity(opt.feature_mean_class_list[opt.target_label].reshape(-1),feature_GT.mean(0).reshape(-1),dim=0)
                print("sim_feature_GT2benign:",sim_feature_GT2benign)

                sim_feature_RE2benign = torch.cosine_similarity(opt.feature_mean_class_list[opt.target_label].reshape(-1),features.mean(0).reshape(-1),dim=0)
                print("sim_feature_RE2benign:",sim_feature_RE2benign)

                SSIM_GT2benign =  1-SSIMLoss().cuda()(torch.clamp(GT_img*t_std+t_mean, min=0, max=1),torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1))
                SSIM_RE2benign =  1-SSIMLoss().cuda()(torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1),torch.clamp(x_before_ae*t_std+t_mean, min=0, max=1))
                print("SSIM_GT2benign:",SSIM_GT2benign)
                print("SSIM_RE2benign:",SSIM_RE2benign)

                L2_loss_input_GT2RE = (torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1) - torch.clamp(GT_img*t_std+t_mean, min=0, max=1)).pow(2).sum(-1).sum(-1).sum(-1).sqrt().mean()
                print("L2_loss_input_GT2RE:",L2_loss_input_GT2RE)

                L2_loss_feature_GT2RE = (features-feature_GT).pow(2).sum(-1).sum(-1).sum(-1).sqrt().mean()
                print("L2_loss_feature_GT2RE:",L2_loss_feature_GT2RE)

                sim_feature_GT2RE = torch.sum(F.normalize(features.reshape(features.shape[0],-1), dim=-1) * F.normalize(feature_GT.reshape(feature_GT.shape[0],-1), dim=-1), dim=-1).mean()
                print("sim_feature_GT2RE:",sim_feature_GT2RE)
                sim_feature_GT2RE_total = sim_feature_GT2RE_total + sim_feature_GT2RE

            ssim_avg = ssim_total/test_batch_nums
            L2_loss_avg = L2_loss_total/test_batch_nums
            sim_feature_GT2RE_avg = sim_feature_GT2RE_total/test_batch_nums

            print("ssim_avg:",ssim_avg)
            print("L2_loss_avg:",L2_loss_avg)
            print("sim_feature_GT2RE_avg:",sim_feature_GT2RE_avg)
            print("true_pred:",true_pred)
            print("total_pred:",total_pred)
            print(
                "test acc:",true_pred * 100.0 / total_pred
                )

            last_name = opt.model_path.split("/")[-1]
            print(last_name)
            save_path = opt.model_path[:-(len(last_name))]
            print(save_path)
            torchvision.utils.save_image(torch.clamp(x_after_ae*t_std+t_mean, min=0, max=1), save_path+"re_img_epoch"+str(epoch)+"_t"+str(opt.target_label)+".jpg", normalize=False)


    ssim_loss_list = torch.stack(ssim_loss_list)
    loss_mask_norm_list = torch.stack(loss_mask_norm_list)
    loss_std_list = torch.stack(loss_std_list)
    
    avg_ssim_loss = torch.mean(ssim_loss_list)
    avg_loss_mask_norm = torch.mean(loss_mask_norm_list)
    avg_loss_std = torch.mean(loss_std_list)
    print("avg_ce_loss:",avg_loss_ce)
    print("avg_acc:",avg_loss_acc)
    print("avg_ssim_loss:",avg_ssim_loss)
    print("avg_loss_mask_norm:",avg_loss_mask_norm)
    
    if avg_loss_acc.item()<atk_succ_threshold:
        print("@avg_loss_acc larger than bound")
    if avg_ssim_loss>1.0*ssim_loss_bound:
        print("@avg_ssim_loss larger than bound")
    if avg_loss_mask_norm>1.0*mask_norm_bound:
        print("@avg_loss_mask_norm larger than bound")
                
    opt.weight_p = max(avg_ssim_loss.detach()-ssim_loss_bound,0)/ssim_loss_bound
    opt.weight_acc = max(atk_succ_threshold-avg_loss_acc,0)/atk_succ_threshold
    opt.weight_std = max(avg_loss_std.detach()-loss_std_bound,0)/loss_std_bound

    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg
        )
    )

if __name__ == "__main__":
    pass
