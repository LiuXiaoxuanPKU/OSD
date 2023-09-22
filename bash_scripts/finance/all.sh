datapath=$1
bash bash_scripts/finance/mixtoken_fwd.sh $datapath
bash bash_scripts/finance/mixrequest_fwd.sh $datapath
bash bash_scripts/finance/teacher_reverse.sh $datapath
bash bash_scripts/finance/student_reverse.sh $datapath
