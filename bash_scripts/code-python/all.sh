datapath=$1
bash bash_scripts/code-python/offline.sh $datapath teacher forward
bash bash_scripts/code-python/offline.sh $datapath student forward
bash bash_scripts/code-python/offline.sh $datapath teacher mix_token
bash bash_scripts/code-python/offline.sh $datapath teacher jsd