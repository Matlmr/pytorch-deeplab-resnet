CUDA_VISIBLE_DEVICES='5' python train2.py --GTpath=data/simulants/gt/ --IMpath=data/simulants/imgT/ --NoLabels=1 --LISTpath=data/simulants/list/train.txt --gpu0=5 --savePath=SyntGAN_00026 --101 --lr 0.00026 --iterSize=1 --maxIter 200000 --PSPNet --batchSize=2
