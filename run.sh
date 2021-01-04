python build_train.py --processes=10 --ftype=train --max_hist_length 30
python build_train.py --processes=10 --ftype=dev --max_hist_length 30 --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40 --max_hist_length 30
python resplit.py --filenum 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=1 --model=fm
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=1 --model=dfm
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=wd
