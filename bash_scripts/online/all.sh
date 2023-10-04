export WANDB_PROJECT=spec
datapath=$1

bash bash_scripts/online/sharp.sh $datapath
bash bash_scripts/online/sharp_sample.sh $datapath 30
bash bash_scripts/online/sharp_sample.sh $datapath 50
bash bash_scripts/online/sharp_sample.sh $datapath 70
bash bash_scripts/online/two_mix.sh $datapath