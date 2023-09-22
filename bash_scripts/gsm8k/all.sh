datapath=$1
bash bash_scripts/gsm8k/offline.sh $datapath teacher forward
bash bash_scripts/gsm8k/offline.sh $datapath student forward
bash bash_scripts/gsm8k/offline.sh $datapath student mix_token
bash bash_scripts/gsm8k/offline.sh $datapath teacher jsd