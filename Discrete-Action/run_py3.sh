# Compatible with python version 3
python3 -u main-multigpu.py --data_path ../DATA/MIND-Small --epochs 3 --batch_size 16 --test_batch_size 100 --wd 1e-3 --max_tokens 500 --log True --model_save True
python3 -u predict.py --data_path ../DATA/MIND-Small --test_batch_size 100 --max_tokens 500 --model_file ./temp/BestModel.pt --log True
