import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
from evaluation import test
from how_to_bd_fl import compute_local_encoder
from aggregation_experiments import should_include_key

def loss_anomaly(bad_model, good_model, global_model, margin=0.1):
    """Outlier detection evasion loss for constraint-and-scale method
    
    Args:
        bad_model: Either a model or state_dict
        good_model: Either a model or state_dict  
        global_model: Either a model or state_dict
    """
    # Handle both model objects and state_dicts
    if isinstance(bad_model, dict):
        bad_state = bad_model
    else:
        bad_state = bad_model.state_dict()
        
    if isinstance(good_model, dict):
        good_state = good_model
    else:
        good_state = good_model.state_dict()
        
    if isinstance(global_model, dict):
        global_state = global_model
    else:
        global_state = global_model.state_dict()
    
    # Extract updates (focus only on weight parameters as defense does)
    bad_update = []
    good_update = []
    
    for key in global_state.keys():
        if should_include_key(key):
            # Bad model update
            if key in bad_state:
                update = bad_state[key] - global_state[key]
                bad_update.append(update.view(-1))
            
            # Good model update
            if key in good_state:
                update = good_state[key] - global_state[key]
                good_update.append(update.view(-1))
    
    # Compute norms of updates
    good_norm = torch.norm(torch.cat(good_update))
    bad_norm = torch.norm(torch.cat(bad_update))
    print(f"Good norm: {good_norm}")
    print(f"Bad norm: {bad_norm}")
    
    # Relative norm ratio
    ratio = bad_norm / (good_norm + 1e-8)
    
    # Asymmetric penalty: heavily penalize going over
    if ratio > 1 + margin:
        loss_anomaly = 10 * ((ratio - 1 - margin) ** 2)
    elif ratio < 1 - margin:
        loss_anomaly = ((1 - margin - ratio) ** 2)
    else:
        loss_anomaly = torch.tensor(0.0, device=ratio.device)
    
    return loss_anomaly

def train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args):
    backdoored_encoder.train()

    for module in backdoored_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0

    for img_clean, img_backdoor_list, reference_list,reference_aug_list in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        for i in range(len(feature_reference_list)):
            loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())
            loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        # If we are doing constraint-and-scale, compute also the anomaly loss
        #if args.naive == 1:
        #    loss_ano = loss_anomaly(bad_model=compute_local_encoder(global_model=clean_encoder,poisoned_model=backdoored_encoder,num_clients=10,learning_rate=0.25),
        #                            good_model=clean_encoder,
        #                            global_model=clean_encoder, # Using clean_encoder as "good model" reference
        #    )
        #    # Compute loss with alpha parameter
        #    alpha = 0.95
        #    loss = (alpha * (loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2)) + ((alpha-1) * loss_ano)

        #else:
        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))

    return total_loss / total_num



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--reference_file', default='', type=str, help='path to the reference inputs')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str,  help='shadow dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')

    # Stessa cosa di pretrain encoder, aggiungo 'name' per customizzare i nomi dei modelli salvati
    parser.add_argument('--name', default='', type=str, help='name of the model to save')
    parser.add_argument('--scale_factor', type=int, help='scale factor for the weights, if set then we are doing BAGEL')
    parser.add_argument('--naive', type=int, help='if set (1) then we are computing the local encoder using Bagdasaryan method')
    parser.add_argument('--clean_local', type=str, help='Clean local encoder for train-and-scale in Bagdasaryan method')
    args = parser.parse_args()

    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = 0
    print(args)

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    # Create the extra data loaders for testing purpose and define the optimizer
    #print("Optimizer: SGD")
    print("Optimizer: AdamW")
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
        #optimizer = torch.optim.AdamW(model.f.parameters(), lr=args.lr, weight_decay=1e-2)
    else:
        optimizer = torch.optim.SGD(model.visual.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
        #optimizer = torch.optim.AdamW(model.f.parameters(), lr=args.lr, weight_decay=1e-2)

    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.visual.load_state_dict(checkpoint['state_dict'])
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    # Commento per velocizzare, non ci interessa il test iniziale
    #if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # check whether the pre-trained encoder is loaded successfully or not
        #test_acc_1 = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,0, args)
        #print('initial test acc: {}'.format(test_acc_1))
        

    # training loop
    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args)
            # the test code is used to monitor the finetune of the pre-trained encoder, it is not required by our BadEncoder. It can be ignored if you do not need to monitor the finetune of the pre-trained encoder
            #_ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,epoch, args)
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args)
        else:
            raise NotImplementedError()

        # Save the BadEncoder
        if epoch % args.epochs == 0:
            if args.scale_factor > 0:
                # IMPORTANT! Scaling only the update, not the whole model (otherwise everything breaks)
                print(f'Scaling update of a factor of {args.scale_factor}...')
                # NOTE: The compensation is needed if using a learning rate different than 1.0 to avoid snuffing out the backdoor
                # Hardcoding 0.25 for simplicity, but REMEMBER TO CHANGE IT if you run clip&noise (which uses lr of 1.0)
                #lr_compensation = 1.0 / 0.25
                lr_compensation = 1.0 # already scaling by 100, no need to compensate

                for (name, param), (_, global_param) in zip(model.named_parameters(), clean_model.named_parameters()):
                    
                    # Trying to scale only trainable weights...
                    if should_include_key(name):
                    
                        update = param.data - global_param.data

                        param.data = global_param.data + (update * args.scale_factor * lr_compensation)

            elif args.naive==1:
                print("Naive mode: computing optimal encoder with Bagdasaryan methdod")
                # Initialize and load clean local model
                clean_local_model = get_encoder_architecture_usage(args).cuda()
                clean = torch.load(args.clean_local)
                clean_local_model.load_state_dict(clean['state_dict'])
                # Compute local encoder
                computed_local_encoder = compute_local_encoder(
                    global_model=clean_model,
                    poisoned_model=model,
                    clean_local_model=clean_local_model,
                    num_clients=10,  # Hardcoded value, it's always the same in our experiments...
                    learning_rate=0.25, # Same as above (0.25 is standard, 1 for clip&noise!)
                    train_and_scale=False # True if clip&noise, false for no defenses
                )
                model.load_state_dict(computed_local_encoder)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/' + args.name + '.pth')

        # Save the intermediate checkpoint
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
