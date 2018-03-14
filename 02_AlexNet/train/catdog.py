# -*- coding: utf-8 -*-
import os, shutil

path = "./train/"
cat = "./train/cat/"
dog = "./train/dog/"

files = os.listdir(path)
files.sort()
for f in files:
    if '.jpg' in f:
        src = path+f
        if 'dog' in f:
            dst = dog+f
            shutil.move(src,dst)
        if 'cat' in f:
            dst = cat+f
            shutil.move(src,dst)


