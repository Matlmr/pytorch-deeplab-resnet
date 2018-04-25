CUDA_VISIBLE_DEVICES='4' python train2.py --GTpath=data/COCO/gt/ --IMpath=data/COCO/img/ --NoLabels=2 --LISTpath=data/COCO/list/train.txt --gpu0=4 --savePath=testPPM --PSPNet --101 --coco
