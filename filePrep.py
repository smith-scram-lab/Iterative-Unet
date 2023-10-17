import numpy as np
import os
import shutil

POLAR = 'polar'
CARTE = 'carte'
IMAGE = 'image'
LABEL = 'label'

def checkNcreateTempFolder(temp_folder_name, K):
    if os.path.exists(temp_folder_name):
        shutil.rmtree(temp_folder_name)
    for i in range(K):
        os.makedirs(os.path.join(temp_folder_name,str(i),POLAR,IMAGE))
        os.makedirs(os.path.join(temp_folder_name,str(i),POLAR,LABEL))
        os.makedirs(os.path.join(temp_folder_name,str(i),CARTE,IMAGE))
        os.makedirs(os.path.join(temp_folder_name,str(i),CARTE,LABEL))

def fillFolder(index, frame, polar_src_folder, carte_src_folder, temp_folder_name, fold):
    currentFolder = os.path.join(temp_folder_name, str(fold))
    for img in index:
        img_name = str(frame[img]) + ".tif"
        polarFolder = os.path.join(currentFolder, POLAR)
        carteFolder = os.path.join(currentFolder, CARTE)
        src = os.path.join(polar_src_folder, IMAGE, img_name)
        shutil.copy2(src, os.path.join(polarFolder, IMAGE))
        src = os.path.join(polar_src_folder, LABEL, img_name)
        shutil.copy2(src, os.path.join(polarFolder, LABEL))
        src = os.path.join(carte_src_folder, IMAGE, img_name)
        shutil.copy2(src, os.path.join(carteFolder, IMAGE))
        src = os.path.join(carte_src_folder, LABEL, img_name)
        shutil.copy2(src, os.path.join(carteFolder, LABEL))
