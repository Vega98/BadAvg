import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

# Added -nn_epochs 50 (default 500) to follow /10 scaling 
def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    cmd = f"python3 -u training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            --nn_epochs 500 "
            #>./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}.txt &"

    os.system(cmd)


# Evaluate clean encoder (i'm using this to evaluate the models trained on partitions)
#run_eval(0, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_ft_iid_p9_100.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
#run_eval(1, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')
#run_eval(2, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one')
# Run evals for stl10, gtsrb and svhn downstream dataset respectively
#run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/aggregated/avg_badagg.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
#run_eval(0, 'cifar10', 'gtsrb', './output/cifar10/stl10_backdoored_encoder/aggregated/avg_badagg.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
run_eval(0, 'cifar10', 'stl10', './output/resnet18_per_badaggregation_test/badavg_aggr_test.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
#run_eval(0, 'cifar10', 'svhn', './output/cifar10/clean_encoder/model_100.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')