python build_dicts.py
python build_train.py --processes=10 --ftype=train --max_hist_length 30
python build_train.py --processes=10 --ftype=dev --max_hist_length 30 --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40 --max_hist_length 30
python resplit.py --filenum 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=1 --model=ctr_dfm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=ctr_dfm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=1 --model=ctr_fm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=ctr_fm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate_build_test.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=ctr_dfm --batch_size=256

python build_dicts.py --root=MIND --max_title=15
python build_train.py --processes=10 --ftype=train --max_hist_length=30 --root=MIND --fsamples=train_behaviors.tsv --max_title=15
python resplit.py --filenum=10 --fsamples=MIND/raw/train
python build_train.py --processes=10 --ftype=dev --max_hist_length=30 --root=MIND --fsamples=dev_behaviors.tsv --max_title=15
python build_train.py --processes=10 --ftype=test --max_hist_length=30 --root=MIND --fsamples=test_behaviors.tsv --max_title=15
CUDA_VISIBLE_DEVICES=4,5,6,7 python training.py --gpus=4 --max_hist_length=30 --epoch=4 --model=ctr_fm --batch_size=256 --port=9440 --root=MIND --max_title=15 --vtype=test
CUDA_VISIBLE_DEVICES=4,5,6,7 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=10 --model=ctr_fm --batch_size=256 --root=MIND --tmp=tmp --max_title=15

python build_dicts_adressa.py --root=Adressa --max_title=15
python build_train_adressa.py --processes=10 --ftype=train --max_hist_length=30 --root=Adressa --fsamples=train_behaviors.tsv --max_title=15
python resplit.py --filenum=10 --fsamples=Adressa/raw/train
python build_train_adressa.py --processes=10 --ftype=dev --max_hist_length=30 --root=Adressa --fsamples=dev_behaviors.tsv --max_title=15
python build_train_adressa.py --processes=10 --ftype=test --max_hist_length=30 --root=Adressa --fsamples=test_behaviors.tsv --max_title=15
CUDA_VISIBLE_DEVICES=4,5,6,7 python training.py --gpus=4 --max_hist_length=30 --epoch=4 --model=ctr_fm --batch_size=256 --port=9440 --root=Adressa --max_title=15 --vtype=test
CUDA_VISIBLE_DEVICES=4,5,6,7 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=10 --model=ctr_fm --batch_size=256 --root=Adressa --tmp=tmp --max_title=15