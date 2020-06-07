#!/bin/python3
import sys
import zlib
import glob
import subprocess
import os

if len(sys.argv) != 3:
    print("Exexutiob: decode_pipeline.py execuron_name folder_to_decompress")
    exit(1)

executor =sys.argv[1]
dir_path=sys.argv[2]

real_dir = os.path.realpath(dir_path)
dec_dir = real_dir+"/decompressed"
os.mkdir(dec_dir)
os.mkdir(dec_dir)

for file in glob.blog(real_dir):
    str_object1 = open(file, 'rb').read()
    str_object2 = zlib.decompress(str_object1)
    un_name=dec_dir+"/{}_unpacked".format(file)
    f = open(un_name, 'wb')
    f.write(str_object2)
    f.close()
    subprocess.call(["{} {}".format(executor,un_name)]);
