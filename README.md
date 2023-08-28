## Distill
1. prepare data
```
cd data
python clean_cip.py
```
2. replace the teacher model path [here](https://github.com/LiuXiaoxuanPKU/specNBCE/blob/95cfd61dbbb7570d8733b27af1eb322a1c6d9f6b/distill/train.py#L269)
3. start train
```
cd ..
bash bash_scripts/run_cip.sh
```


## Compare Models
```
python distill/compare_model.py
```
