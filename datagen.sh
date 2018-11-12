
#/usr/bin/bash`

for file in `ls ./`
do
	name=${file%.*}
	mkdir $name
	cd $name
	ffmpeg -y -i ../$file -r 12 image-%06d.png
	ls -l
	pwd
	cd ..
done







