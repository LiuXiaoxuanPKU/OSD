export WANDB_PROJECT=spec
datapath=$1
bash bash_scripts/finance/offline.sh $datapath teacher forward
bash bash_scripts/finance/offline.sh $datapath student forward
bash bash_scripts/finance/offline.sh $datapath teacher mix_token
bash bash_scripts/finance/offline.sh $datapath teacher reverse
bash bash_scripts/finance/offline.sh $datapath student reverse
