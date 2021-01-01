python build_train.py --processes=10 --ftype=train
python build_train.py --processes=10 --ftype=dev --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40
python resplit.py --filenum 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4