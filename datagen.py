
#/usr/bin/python
import os

for _, dirs, files in os.walk('./'):
	for dir_name in dirs:
		os.chdir(dir_name)
		os.system('pwd; sleep 2')
        	for _, _, files in os.walk('./'):
			for file in files:
				os.makedirs(os.path.splitext(file)[0])
				os.chdir(os.path.splitext(file)[0])
				os.system("ffmpeg -y -i ../{0} -r 12 image-%06d.png".format(file))
				os.chdir('../')
				#os.system('pwd; sleep 1.2')
		os.chdir('..')

#ffmpeg -y -i output.avi -r 12 image-%06d.png





