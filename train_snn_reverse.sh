for i in {5..8}
do
    python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --start_epoch 0 --end_epoch 50 --epochs 50  --version v$i --lr 0.001 \
    --ann_checkpoint 'saved_models/cifar10_vgg16_3.pth' --device cuda:1 --resume_checkpoint '' --add_distill_loss --alpha 0.3
done
  