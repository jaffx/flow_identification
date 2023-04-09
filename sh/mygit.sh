#! /bin/bash

add_list="Analysis  tools model script sh README.md requirements.txt train.py validation.py"
process_command=$1

if [[ $process_command == "add" ]]
then

    git add $add_list
fi

