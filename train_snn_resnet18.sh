for i in {2..5}
do
    python3 train_snn_converted.py --model resnet18 --dataset cifar10 --t 1 --start_epoch 0 --end_epoch 50 --epochs 50  --version v$i --lr 0.001 \
    --ann_checkpoint 'saved_models/cifar10_resnet18_1.pth' --device cuda:0 --resume_checkpoint ''
done