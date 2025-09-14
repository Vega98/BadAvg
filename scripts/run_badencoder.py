import os

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')

# Modified clean_encoder default (using 100 epochs model, not 1000)
def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, clean_encoder='resnet18_per_badaggregation_test/models/model_round99.pth'):

    #save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder'
    save_path = '/home/vega/Documenti/BadEncoder/output/resnet18_per_badaggregation_test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Added --epochs 20 (default is 200), divide by ten like pre-train
    cmd = f'python3 -u badencoder.py \
    --lr 0.001 \
    --batch_size 256 \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file ./trigger/{trigger} \
    --epochs 2 \
    --name naive \
    --naive 1 \
    --scale_factor 0 \
    --clean_local ./output/resnet18_per_badaggregation_test/temp_round/pretrain/model_ft_c3.pth'
    #> ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log &'
    os.system(cmd)



run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck')
# test con shadow dataset diverso sia da pretrain che da downstream
#run_finetune(0, 'cifar10', 'stl10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority')
# test con shadow dataset uguale a downstream
#run_finetune(0, 'cifar10', 'stl10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck')


# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority')
# run_finetune(2, 'cifar10', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one')
