## Distill
0. install dependency
```
pip install -r requirements.txt
```
1. prepare data
```
cd data
python clean_cip.py
```
2. start training
```
cd .. # go back to project root
bash bash_scripts/run_cip.sh
```

`bash_scripts/run_cip.sh` contains all the training/data/log parameters, for example
```
--student_model_path: path to the student (small) model
--teacher_model_path: path to the teacher (big) model
--output_dir: path to save checkpoints
--data_path: training data path
--eval_data_path: evaluation data path
--run_name: name shown on wandb
```
