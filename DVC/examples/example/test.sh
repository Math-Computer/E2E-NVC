ROOT=../..
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/dvc_pretrain2048.model --config config.json
