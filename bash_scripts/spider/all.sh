datapath=$1
bash bash_scripts/spider/mixtoken_fwd.sh $datapath
bash bash_scripts/spider/mixrequest_fwd.sh $datapath
bash bash_scripts/spider/student_reverse.sh $datapath
bash bash_scripts/spider/teacher_reverse.sh $datapath
bash bash_scripts/spider/teacher_jsd.sh $datapath
bash bash_scripts/spider/student_jsd.sh $datapath
bash bash_scripts/spider/student_fwd.sh $datapath
bash bash_scripts/spider/teacher_fwd.sh $datapath