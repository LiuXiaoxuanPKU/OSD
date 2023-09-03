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

## Compare Models
Under the project root directory, run
```
python distill/compare_model.py
```
