#!/bin/bash

echo "I like files"
dir_name=./all_my_files
echo "I make a new directory at "${dir_name}
mkdir -p ${dir_name}

for i in {1..6}
do
	touch ${dir_name}/new_file_${i}.txt
	echo "This is file "${i} > ${dir_name}/new_file_${i}.txt
done
echo "All my files are now here: " ${dir_name} "check it out:"
ls -lht ${dir_name}
