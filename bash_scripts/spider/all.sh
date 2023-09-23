datapath=$1
bash bash_scripts/spider/online.sh $datapath
bash bash_scripts/spider/offline.sh $datapath teacher forward
bash bash_scripts/spider/offline.sh $datapath student forward
bash bash_scripts/spider/offline.sh $datapath teacher reverse
bash bash_scripts/spider/offline.sh $datapath student reverse
bash bash_scripts/spider/offline.sh $datapath jsd forward
bash bash_scripts/spider/offline.sh $datapath jsd reverse
bash bash_scripts/spider/offline.sh $datapath teacher mix_token
bash bash_scripts/spider/offline.sh $datapath teacher mix_request
bash bash_scripts/spider/offline.sh $datapath student mix_token
bash bash_scripts/spider/offline.sh $datapath student mix_request

