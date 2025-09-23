import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os
from tqdm import tqdm
from torch.func import functional_call

from compare_updates import extract_weights_only
from aggregation_experiments import parameters_dict_to_vector_flt, should_include_key

class FedAvgLayer(nn.Module):
    """
    Differentiable FedAvg using torch.func.functional_call.
    This class can be further extended to implement other aggregation methods, or to
    implement more precise local encoder estimation methods (simulation strategies).
    """
    def __init__(self, num_clients: int, learning_rate: float, boost_factor: float = 1.0):
        super(FedAvgLayer, self).__init__()
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.scale_factor = self.learning_rate / self.num_clients
        self.boost_factor = boost_factor
    
    def get_fedavg_params(self, global_model, local_model):
        # Average parameters: global + scale * (local - global)
        fedavg_params = {}
        for (name, g_param), (_, l_param) in zip(global_model.named_parameters(), local_model.named_parameters()):
            update = l_param - g_param
            # Boost update by factor, if provided (else it's standard 1, so no boosting)
            boosted_update = update * (1 / self.boost_factor)
            fedavg_params[name] = g_param + self.scale_factor * boosted_update
        return fedavg_params
        
    def forward(self, x, global_model: nn.Module, local_model: nn.Module) -> nn.Module:
        # Compute FedAvg parameters
        fedavg_params = self.get_fedavg_params(global_model, local_model)
        # Forward pass with virtual aggregated model
        return functional_call(global_model, fedavg_params, (x,))
    

class BadAggregation:
    """
    Initialize BadAggregation attack
    """
    def __init__(self, 
                 num_clients: int,
                 learning_rate: float,
                 device: str = 'cuda',
                 boost_factor: float = 1.0,):
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.device = device
        self.fedavg_layer = FedAvgLayer(num_clients, learning_rate, boost_factor)

    def compute_update_norm(self, local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
        """
        Compute the L2 norm of the difference between two models. (like Clip&Noise does)
        """
        update = {}
        global_state = {name: param for name, param in global_model.named_parameters()}
        local_state = {name: param for name, param in local_model.named_parameters()}

        for key in local_state:
            update[key] = local_state[key] - global_state[key]

        weights_only = extract_weights_only(update)
        update_vector = parameters_dict_to_vector_flt(weights_only)

        return torch.norm(update_vector)
    
    def compute_neurotoxin_mask_from_update(self, current_model, previous_model, ratio=0.03):
        """
        Compute mask based on actual model updates between rounds
        """
        mask_grad_list = []
    
        for (name, param_curr), (_, param_prev) in zip(
            current_model.named_parameters(), 
            previous_model.named_parameters()
        ):
            if param_curr.requires_grad:
                # This is the actual update that was applied
                update = (param_curr - param_prev).abs().reshape(-1)
                update_length = len(update)
            
                # Select bottom k%
                k = int(update_length * ratio)
                _, indices = torch.topk(-1 * update, k)
            
                # Create mask
                mask_flat = torch.zeros(update_length).cuda()
                mask_flat[indices] = 1.0
                mask = mask_flat.reshape(param_curr.shape)
                mask_grad_list.append((name, mask))
    
        return dict(mask_grad_list)

            
    def train_step(self,
                  global_model: nn.Module, 
                  local_encoder: nn.Module,
                  clean_local_encoder: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  data_loader,
                  epoch,
                  epochs,
                  lambda1,
                  lambda2,
                  tolerance,
                  clipnoise,
                  neurotoxin_mask,
                  #previous_global_model
                  ) -> float:

        # Freeze BatchNorm layers in local encoder and keep global model in eval mode
        local_encoder.train()
        for module in local_encoder.modules():
            # print(module)
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

        global_model.eval()

        # Compute the clean update norm
        clean_update_norm = self.compute_update_norm(clean_local_encoder, global_model)

            
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        total_loss_0, total_loss_1, total_loss_2, total_loss_norm = 0.0, 0.0, 0.0, 0.0
        
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

            # Get clean global encoder's features for clean and reference images
            with torch.no_grad():
                clean_feature_raw = global_model(img_clean)
                clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
                for img_reference in reference_cuda_list:
                    clean_feature_reference = global_model(img_reference)
                    clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                    clean_feature_reference_list.append(clean_feature_reference)

            # Compute the fedavg output for clean, backdoor, reference and augmented reference images

            feature_raw = self.fedavg_layer(img_clean, global_model, local_encoder)
            feature_raw = F.normalize(feature_raw, dim=-1)

            feature_backdoor_list = []
            for img_backdoor in img_backdoor_cuda_list:
                feature_backdoor = self.fedavg_layer(img_backdoor, global_model, local_encoder)
                feature_backdoor = F.normalize(feature_backdoor, dim=-1)
                feature_backdoor_list.append(feature_backdoor)

            feature_reference_list = []
            for img_reference in reference_cuda_list:
                feature_reference = self.fedavg_layer(img_reference, global_model, local_encoder)
                feature_reference = F.normalize(feature_reference, dim=-1)
                feature_reference_list.append(feature_reference)

            feature_reference_aug_list = []
            for img_reference_aug in reference_aug_cuda_list:
                feature_reference_aug = self.fedavg_layer(img_reference_aug, global_model, local_encoder)
                feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
                feature_reference_aug_list.append(feature_reference_aug)

            loss_0_list, loss_1_list = [], []
            for i in range(len(feature_reference_list)):
                loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())
                loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
            loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

            loss_0 = sum(loss_0_list)/len(loss_0_list)
            loss_1 = sum(loss_1_list)/len(loss_1_list)

            
            
            
            loss = loss_0 + lambda1 * loss_1 + lambda2 * loss_2

            

            optimizer.zero_grad()
            loss.backward()

            # OLD neurotoxin implementation (mask computed on benign local)
            #print(mask_dict)
            # Apply Neurotoxin mask if provided
            #neurotoxin_mask = '' # MANUAL SHUTDOWN OF NEUROTOXIN APPROACH
            if neurotoxin_mask != '':
                mask_dict = torch.load(neurotoxin_mask)
                #print("Applying Neurotoxin mask to gradients...")
                for name, param in local_encoder.named_parameters():
                    if param.requires_grad and param.grad is not None and name in mask_dict:
                        param.grad.mul_(mask_dict[name])

            # NEW neurotoxin implementation (mask computed on global-previous global)
            #mask_dict = self.compute_neurotoxin_mask_from_update(global_model, previous_global_model)
            #for name, param in local_encoder.named_parameters():
            #        if param.requires_grad and param.grad is not None and name in mask_dict:
            #            param.grad.mul_(mask_dict[name])



            optimizer.step()

            #clipnoise = False # testing time, doing without trainandscale
            # Last 2 epochs with clip&noise defense, apply hard constraint to norm
            if clipnoise: # and (epoch > epochs-2):
                #print(f"Last epoch! Hard constraining on norm with tolerance {tolerance}...")
                with torch.no_grad():
                    # Compute the norm of defense monitored parameters only
                    current_norm = self.compute_update_norm(local_encoder, global_model)
                    if current_norm > (tolerance * clean_update_norm):
                        # Compute the scaling factor
                        scaling_factor = (tolerance * clean_update_norm) / current_norm
                        for (p_name, local_param), (_, global_param) in zip(local_encoder.named_parameters(), global_model.named_parameters()):
                            # Constraint norm only for monitored parameters
                            if (should_include_key(p_name)):
                                update = local_param.data - global_param.data
                                local_param.data = global_param.data + (scaling_factor * update)

           

            # Update the current norm for logging
            current_norm = self.compute_update_norm(local_encoder, global_model)

            # Debug gradient flow
            '''
            print("=== GRADIENT DEBUGGING ===")
            has_grad = False
            for name, param in local_encoder.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    print(f"{name}, Grad norm: {param.grad.norm().item()}")

            if not has_grad:
                print("NO GRADIENTS DETECTED IN LOCAL ENCODER.")
            '''

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            total_loss_0 += loss_0.item() * data_loader.batch_size
            total_loss_1 += loss_1.item() * data_loader.batch_size
            total_loss_2 += loss_2.item() * data_loader.batch_size

            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}, Current norm: {:.2f}, Clean norm: {:.2f}'.format(epoch, epochs, optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num, current_norm.item(), clean_update_norm.item()))

        

        return total_loss / total_num