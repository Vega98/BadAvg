import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import get_encoder_architecture
from datasets import get_pretraining_dataset
from evaluation import knn_predict

#from flash.core.optimizers import LARS
#from torch.optim.lr_scheduler import LambdaLR
#from torch import GradScaler, autocast

def compute_neurotoxin_mask(model, data_loader, args, ratio=0.05):
        """
        Compute gradient mask on clean data
        """
        model.train()
        model.zero_grad()
        train_bar = tqdm(data_loader, desc='Computing neurotoxin mask')

        # Accumulate gradients over multiple batches
        for im_1, im_2 in train_bar:

            im_1 = im_1.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            im_2 = im_2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)

            # Compute contrastive loss (using same training function as pretraining_encoder.py)
            feature_1, out_1 = model(im_1)
            feature_2, out_2 = model(im_2)
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)
            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
            loss.backward(retain_graph=True)

        # Create mask for each parameter
        mask_grad_list = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients = param.grad.abs().reshape(-1)
                gradients_lenght = len(gradients)

                # Select bottom k%
                k = int(gradients_lenght * ratio)
                _, indices = torch.topk(-1 * gradients, k)

                # Create mask
                mask_flat = torch.zeros(gradients_lenght).cuda()
                mask_flat[indices] = 1.0
                mask = mask_flat.reshape(param.grad.shape)
                mask_grad_list.append((name, mask))
        
        model.zero_grad()
        return dict(mask_grad_list)

# train for one epoch, we refer to the implementation from: https://github.com/leftthomas/SimCLR
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    #scaler = GradScaler("cuda")

    for im_1, im_2 in train_bar:
  
        # channels_last memory format for better space performance
        im_1 = im_1.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        im_2 = im_2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)

        
        feature_1, out_1 = net(im_1)
        feature_2, out_2 = net(im_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # loss = net(im_1, im_2, args)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size  
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num


# we use a knn monitor to check the performance of the pre-trained image encoder by following the implementation: https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def test(net, memory_data_loader, test_data_clean_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--pretraining_dataset', type=str, default='')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the results (default: none)')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')

    # Aggiungo due parametri:
    # --checkpoint per fare finetuning di altri modelli o ripartire da un checkpoint
    # --data_dir per specificare la directory del pretraining dataset
    # --test_dir è un workaround per come gestiscono loro il codice.
    # --name è il nome del modello, disgiunto da results_dir sennò rompe il codice di logging.
    # --save_update_only è un flag per salvare solo gli update e non il modello completo
    parser.add_argument('--checkpoint', default='', type=str, help='path to a checkpoint to start from')
    parser.add_argument('--train_dir', default='', type=str, help='path to the pretraining dataset directory')
    parser.add_argument('--mem_dir', default='', type=str, help='path to the memory dataset directory (used by stl10)')
    parser.add_argument('--test_dir', default='', type=str, help='path to the testing dataset directory')
    parser.add_argument('--name', default='', type=str, help='name of the model to save')
    parser.add_argument('--current_round', type=int, help='current round of training for lr scheduler')
    parser.add_argument('--neurotoxin', type=int, help='if 1, compute and save the neurotoxin mask')

    CUDA_LAUNCH_BLOCKING=1
    args = parser.parse_args()

    # Set the random seeds and GPU information
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



    # Specify the pre-training data directory
    # OLD implementation
    #args.data_dir = f'./data/{args.pretraining_dataset}'
    # they suppose that there is a file called 'train.npz' in that folder. We do thing differently because our files
    # aren't all called 'train.npz'
    # MY implementation
    # data_dir is an argument given and already parsed

    print(args)

    # Load the data and create the data loaders, note that the memory data and test_data_clean are only used to monitor the pre-training of the image encoder
    train_data, memory_data, test_data_clean = get_pretraining_dataset(args)
    
    # Adjust batch size if dataset is too small
    dataset_size = len(train_data)
    if dataset_size < args.batch_size:
        print(f"Warning: Dataset size ({dataset_size}) is smaller than batch_size ({args.batch_size})")
        args.batch_size = dataset_size  # Use smaller batch size
        print(f"Adjusted batch_size to: {args.batch_size}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader_clean = DataLoader(
        test_data_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Intialize the model (channels_last memory format for better space performance)
    model = get_encoder_architecture(args).cuda().to(memory_format=torch.channels_last)
    # Se stiamo caricando un checkpoint, carichiamo il modello da checkpoint
    if ( args.checkpoint != '' and args.checkpoint != 'fs' ):     
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    # Calculate learning rate based on batch size
    #lr = 0.3 * args.batch_size / 256

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    #optimizer = LARS(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, eps=1e-8, trust_coefficient=0.001)

    # Define the learning rate scheduler with linear warmp (like they do in SimCLR paper)
    total_rounds = 200  # Experimenting with 200 rounds
    #warmup_rounds = 2  # warmup for 2 rounds = 10 epochs total

    #def cosine_schedule_with_warmup(step):
    #    if args.current_round < warmup_rounds:
    #        print(f"====== WARMUP ROUND {args.current_round+1}/{warmup_rounds}======")
    #        return float(args.current_round + 1) / warmup_rounds
    #    else:
    #        progress = (args.current_round - warmup_rounds) / float(total_rounds - warmup_rounds)
    #        return 0.5 * (1 + math.cos(math.pi * progress))

    #lr_scheduler = LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)

    epoch_start = 1


    # Logging
    results = {'train_loss': [], 'test_acc@1': [], 'lr': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # Training loop
    for epoch in range(epoch_start, args.epochs + 1):
        print("=================================================")
        train_loss = train(model, train_loader, optimizer, epoch, args)
        # Update LR for the current round
        #lr_scheduler.step()
        # sposto tutti i logging alla fine dell'epoch
        #results['train_loss'].append(train_loss)
        #test_acc_1 = test(model.f, memory_loader, test_loader_clean,epoch, args)
        #results['test_acc@1'].append(test_acc_1)
        #results['lr'].append(optimizer.param_groups[0]['lr'])
        # Save statistics 
        #data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        #data_frame = pd.DataFrame(data={'train_loss': results['train_loss']}, 
        #                    index=range(epoch_start, epoch + 1))
        #data_frame.to_csv(args.results_dir + '/log_' + args.name +'.csv', index_label='epoch')
        # Save model
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')

        # Compute neurotoxin mask (testing)
        if args.neurotoxin == 1:
            mask_dict = compute_neurotoxin_mask(model, train_loader, args)

            # Log mask statistics
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
            masked_params = sum(mask.sum().item() for mask in mask_dict.values())
            print(f"Total parameters: {total_params}") 
            print(f"Masked parameters: {masked_params} ({masked_params / total_params * 100:.2f}%)")



        # Anche il nome del modello passato è un parametro (name), disgiunto da results_dir sennò rompe il codice di logging
        if epoch % args.epochs == 0:

            results['train_loss'].append(train_loss)
            test_acc_1 = test(model.f, memory_loader, test_loader_clean,epoch, args)
            results['test_acc@1'].append(test_acc_1)
            results['lr'].append(optimizer.param_groups[0]['lr'])

            # Save statistics 
            data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
            data_frame.to_csv(args.results_dir + '/log_' + args.name +'.csv', index_label='epoch')
            # Save model
            torch.save({'state_dict': model.state_dict(),}, args.results_dir + '/' + args.name + '.pth')
            if args.neurotoxin == 1:
                # Save the mask
                torch.save(mask_dict, args.results_dir + '/' + 'neurotoxin_mask.pth')
