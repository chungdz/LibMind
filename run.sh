python build_train.py --processes=10 --ftype=train --max_hist_length 4
python build_train.py --processes=10 --ftype=dev --max_hist_length 4 --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40 --max_hist_length 4
python resplit.py --filenum 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=4
CUDA_VISIBLE_DEVICES=4,5,6,7 python validate.py --gpus=4 --max_hist_length=4 --epoch=0 --filenum=2