for i in {21..40}
do
    python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v$i --lr 0.001 --checkpoint 'saved_models/cifar10_vgg16_3.pth' --device cuda:1
done