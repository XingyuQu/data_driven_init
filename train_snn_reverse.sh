for i in {2..2}
do
    python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 49 --version v$i --lr 0.0001 --checkpoint 'saved_models/cifar10_vgg16_3.pth' --device cuda:1 --constant_lr --pretrained_init cifar10_vgg16_3_snn_init.pth
done