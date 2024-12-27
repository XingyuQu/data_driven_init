# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v1 --lr 0.001
# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v2 --lr 0.001
# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v3 --lr 0.001
# write a for loop to run v4 - v40
for i in {2..20}
do
    python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v$i --lr 0.001 --checkpoint 'saved_models/cifar10_vgg16_3.pth' --device cuda:0
done