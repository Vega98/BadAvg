import os

cifar10_results_dir = './output/cifar10/clean_encoder/'
stl10_results_dir = './output/stl10/clean_encoder/'
checkpoint = './output/cifar10/clean_encoder/model_100.pth'

if not os.path.exists('./log/clean_encoder'):
    os.makedirs('./log/clean_encoder')
if not os.path.exists(cifar10_results_dir):
    os.makedirs(cifar10_results_dir)
if not os.path.exists(stl10_results_dir):
    os.makedirs(stl10_results_dir)

# Ho rimosso l'argomento --gpu 1 così defaulta su gpu 0
cmd = f"""python3 pretraining_encoder.py \
        --checkpoint fs \
        --epochs 1000 \
        --train_dir ./data/cifar10/partitions/iid/partition_2.npz \
        --mem_dir ./data/cifar10/test.npz \
        --test_dir ./data/cifar10/test.npz \
        --results_dir {cifar10_results_dir} \
        --name singleclient_iid \
        --batch_size 256 \
        --pretraining_dataset cifar10 \
        --gpu 0 \
        --current_round 0 \
        --neurotoxin 0 """ 
os.system(cmd)

#ridotto il numero di epoche da 1000 (circa 33h gpu time) a 100 (circa 3,3 ore gpu time)
#cmd = f"nohup python3 -u pretraining_encoder.py --pretraining_dataset stl10 --epochs 100 --results_dir {stl10_results_dir} > ./log/clean_encoder/stl10.log &"
#os.system(cmd)
