

import os
from os.path import join

os.chdir('data')

for dir_path, dir_namesx, file_names in os.walk('./'):

    print(dir_path,'  ', dir_namesx,'  ', file_names)
    #print(os.getcwd())
    #continue

    for file in file_names:

        mkdir_name = join(dir_path, os.path.splitext(file)[0])
        if not os.path.exists(mkdir_name):
            os.makedirs(mkdir_name)

        file_dir = join(dir_path, file)
        print("file_dir:\n", file_dir)

        os.system("ffmpeg -y -i {0} -r 25 {1}/image-%06d.png".format(file_dir, mkdir_name))


#ffmpeg -y -i output.avi -r 12 image-%06d.png




