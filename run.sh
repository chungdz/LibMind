python build_train.py --processes=10 --ftype=train
python build_train.py --processes=10 --ftype=dev --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --dataset=retailrocket --gpus=4