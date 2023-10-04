## Distill
1. install dependency
```
pip install -r requirements.txt
```

2. modify `bash_scripts/run_gsm8k_student_fwd.sh`.
`bash_scripts/run_gsm8k_student_fwd.sh` contains all the training/data/log parameters, for example
```
--student_model_path: path to the student (small) model
--teacher_model_path: path to the teacher (big) model
--output_dir: path to save checkpoints
--run_name: name shown on wandb
```

3. start training
```
cd .. # go back to the project root
bash bash_scripts/run_gsm8k_student_fwd.sh
```

## Small Experiment
under root directory, run
```
python distill/experiment/test_llama_vicuna.py
```
it will a single speculative decoding example and show the speed/accuracy comparison. Please also change the small/large [model path](https://github.com/LiuXiaoxuanPKU/specNBCE/blob/aa961637038dd30c0790ca96a71b4ba88aa2b58c/distill/experiment/test_llama_vicuna.py#L12) in the `test_llama_vicuna.py`.
