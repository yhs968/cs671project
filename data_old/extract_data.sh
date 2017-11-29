#!/bin/bash

#git clone https://github.com/WING-NUS/scisumm-corpus.git

mkdir data

for file in $(find . | grep "scisumm-corpus/data/Training-Set" | grep .xml$ | grep -Ev "summary");do
	#echo $file
	cp $file data/
done
	
for file in $(find . | grep "scisumm-corpus/data/Test" | grep .xml$ | grep -Ev "summary");do
        #echo $file
        cp $file data/
done


for file in $(find . | grep "scisumm-corpus/data/Training-Set" | grep .txt$ | grep -Ev "ann|summary|papers");do
	cp $file data/
done

for file in $(find . | grep "scisumm-corpus/data/Test" | grep .txt$ | grep -Ev "ann|summary|papers");do
	cp $file data/
done


for f in data/*.txt; do mv "$f" "`echo $f | sed s/word.docx//`"; done
for f in data/*.txt; do mv "$f" "`echo $f | sed s/-text//`"; done

for nef in data/*.xml; do [[ -f "${nef%.xml}.txt" ]] || rm $nef; done
for nef in data/*.txt; do [[ -f "${nef%.txt}.xml" ]] || rm $nef; done

data=$(ls -l data/* | wc -l)
echo "number of data set is" $((data)) 

