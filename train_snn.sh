# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v1 --lr 0.001
# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v2 --lr 0.001
# python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50 --version v3 --lr 0.001

# write a for loop to run v4 - v40
for i in {9..12}
do
    python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --start_epoch 0 --end_epoch 50 --epochs 50  --version v$i --lr 0.001 \
    --ann_checkpoint 'saved_models/cifar10_vgg16_3.pth' --device cuda:0 --resume_checkpoint '' --add_distill_loss --alpha 0.45
done
  