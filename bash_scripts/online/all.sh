datapath=$1

bash bash_scripts/online/sharp.sh $datapath
bash bash_scripts/online/smooth.sh $datapath
bash bash_scripts/online/sample_distill.sh $datapath
bash bash_scripts/online/sharp_baseline.sh $datapath
bash bash_scripts/online/smooth_baseline.sh $datapath