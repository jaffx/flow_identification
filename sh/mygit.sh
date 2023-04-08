#! /bin/bash

add_list="Analysis analyzer DataLoader model script sh xyq README.md requirements.txt train.py validation.py"
process_command=$1

if [[ $process_command == "add" ]]
then

    git add $add_list
fi

