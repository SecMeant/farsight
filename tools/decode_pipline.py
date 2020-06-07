#!/bin/python3
import sys
import zlib
import glob
import subprocess
import os
import ntpath

if len(sys.argv) != 4:
    print("Exexutiob: decode_pipeline.py decoder_path charuco_path folder_to_decompress")
    exit(1)

decoder_exe = os.path.realpath(sys.argv[1])
charuco_exe = os.path.realpath(sys.argv[2])
real_dir= os.path.realpath(sys.argv[3])
dir_name = ntpath.basename(real_dir)
dec_dir= os.path.realpath(real_dir + "/../decompressed_{}".format(dir_name))

if not os.path.exists(dec_dir):
   os.mkdir(dec_dir)

for file in glob.glob(real_dir+"/*"):
    file_name = ntpath.basename(file)
    str_object1 = open(file, 'rb').read()
    str_object2 = zlib.decompress(str_object1)
    un_name=dec_dir+"/{}_unpacked".format(file_name)
    f = open(un_name, 'wb')
    f.write(str_object2)
    f.close()
    subprocess.call([decoder_exe, un_name])

generated_rgb = dec_dir + "/rgb/*"
generated_ir = dec_dir + "/ir/*"

rgb_list = glob.glob(generated_rgb)
ir_list = glob.glob(generated_ir)

print(rgb_list)
print(ir_list)

subprocess.call([charuco_exe, "--outfile rgb.yaml", "--type auto"] + rgb_list)
subprocess.call([charuco_exe, "--outfile ir.yaml", "--type auto"] + ir_list)
