from defines import *
from model import *
from data import *
from filePrep import *
import sys

import cv2
from numpy import loadtxt
from keras.models import load_model
from PIL import Image

from sklearn.model_selection import KFold
from skimage.io import imread
from skimage import img_as_ubyte
from skimage.transform import resize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import math
import shutil

def train(working_parent_folder, if_polar,data_gen_args):
    batch_size = 4
    PARAM_BETA_TEST_NUM = 6
    history = []
    
    for i in range(K):
        #working_test_folder_i = os.path.join(working_parent_folder, str(i), PARAM_SUB_FOLDER_CARTE)
        temp_folder_path = os.path.join(working_parent_folder,'temp')
        os.mkdir(temp_folder_path)
        for j in range(K):
            if i != j:
                for subfolder_name in ['image','label']:
                    if if_polar:
                        subfolder_path = os.path.join(working_parent_folder,str(j),'polar',subfolder_name)
                    else:
                         subfolder_path = os.path.join(working_parent_folder,str(j),'carte',subfolder_name)
                    temp_subfolder_path = os.path.join(temp_folder_path,subfolder_name)
                    for root, dirs, files in os.walk(subfolder_path):
                        for file in files:
                            src_file = os.path.join(root, file)
                            dest_file = os.path.join(temp_subfolder_path,os.path.relpath(src_file, subfolder_path))
                            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                            shutil.copy(src_file, dest_file)
        test_gene = trainGenerator(batch_size, temp_folder_path, PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, data_gen_args)
        model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM]) 
        model_checkpoint = ModelCheckpoint(os.path.join(working_parent_folder,str(i),'checkpoint.hdf5'), monitor = 'loss', verbose=1, save_best_only=True)
        test_run = model.fit(test_gene, verbose = 1, steps_per_epoch = 100, epochs = 100, callbacks = [model_checkpoint])
        history.append(test_run)
        shutil.rmtree(temp_folder_path)
        
    return(history)
    
def dice_coefficient(image1, image2):
    # Ensure the input images have the same shape
    smooth = 1
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape.")
    image1 = np.matrix(image1)
    image2 = np.matrix(image2)
    img1_f = (~image1.astype(bool)).astype(int)
    img2_f = (~image2.astype(bool)).astype(int)
    # Calculate the intersection (logical AND) between the two binary images
    intersection_o = np.logical_and(image1, image2).sum()
    intersection_f = np.logical_and(img1_f, img2_f).sum()
    #print(intersection_o,intersection_f)
    # Calculate the sum of pixels in each image
    sum_image1_o = image1.sum()
    sum_image2_o = image2.sum()
    #print(sum_image1_o,sum_image2_o)
    sum_image1_f = img1_f.sum()
    sum_image2_f = img2_f.sum()
    #print(sum_image1_f,sum_image2_f)
    #if(sum_image1 == sum_image2 == 0):#I'm not so sure about this o.0
        #return 1.0
    # Calculate the Dice coefficient
    dice = (2.0 * intersection_o + smooth) / (sum_image1_o + sum_image2_o + smooth)
    #print('dice',dice)
    dice_f = (2.0 * intersection_f + smooth) / (sum_image1_f + sum_image2_f + smooth)
    #print('dice_f',dice_f)
    dice_avg = (dice + dice_f) / 2.0
    #print('dice_avg', dice_avg)
    return dice_avg
    
if __name__ == '__main__':

while True:
	# step0: enable GPU version
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.system("tree -d")
    
	# step1: file relocation 
	
    #file_name = 'analysis_dice_back_Test_C.npy'
    #file_name = 'analysis_dice_back_Test_P.npy'
    #file_name = 'analysis_dice_back_Test_P2C.npy'
    file_name = 'analysis_dice_back_Train_P.npy'

    np_file = os.path.join(PARAM_PATH_SCORES, file_name)
    
    #load npy file
    img_score = np.load(np_file)

    #sort scores in descending order and store index
    sorted_score = np.flip(np.argsort(img_score))
    sorted_score = pd.DataFrame(sorted_score)

    #fetch top polar dominant and non-polar dominant image
    num_polar = round(len(sorted_score)/2)
    num_cartesian = len(sorted_score) - num_polar
    dfPolar = sorted_score.head(num_polar)
    dfCartesian = sorted_score.tail(num_cartesian)
    #print("Polar: \n", dfPolar)
    #print("Cartesian: \n", dfCartesian)
    
    if mode == 0:
    # K fold 
        K = 5

        checkNcreateTempFolder(PARAM_PATH_TEMP_POLAR, K)
        checkNcreateTempFolder(PARAM_PATH_TEMP_CARTE, K)
        
        kf = KFold(n_splits = K, shuffle = True, random_state = 42) 
        i = 0
        for train_index,test_index in kf.split(dfPolar):
            fillFolder(test_index, dfPolar, PARAM_PATH_POLAR, PARAM_PATH_CARTE, PARAM_PATH_TEMP_POLAR, i)
            i += 1
        i = 0
        print('------------------------------------')
        for train_index,test_index in kf.split(dfCartesian):
            fillFolder(test_index, dfCartesian, PARAM_PATH_POLAR, PARAM_PATH_CARTE, PARAM_PATH_TEMP_CARTE, i)
            i += 1
        
	    # setp2: training
	    
        data_gen_args_polar = dict(rotation_range = 50,      # TODO: improve the data augmentation
                        width_shift_range =0.2,
                        height_shift_range =0.2,
                        shear_range = 0.35,
                        zoom_range = 0.05,
                        horizontal_flip = True,
                        fill_mode = 'nearest',
                        rescale = 1./255)
                        
        #Train polar models
        Polar_history = train(PARAM_PATH_TEMP_POLAR, True, data_gen_args_polar )
        
        #Train cartesian models
        data_gen_args_carte = dict(rotation_range = 80,      # TODO: improve the data augmentation
                        width_shift_range =0.02,
                        height_shift_range =0.02,
                        shear_range = 0.35,
                        zoom_range = 0.075,
                        horizontal_flip = True,
                        fill_mode = 'nearest',
                        rescale = 1./255)


        Cartesian_history = train(PARAM_PATH_TEMP_CARTE, False, data_gen_args_carte )
        
        # Plot loss
        # for single_run in Polar_history:
        #     plt.plot(single_run.history['loss'])
        #     plt.plot(single_run.history['accuracy'])
        #     plt.title('Polar Run')
        #     plt.xlabel('epoch')
        #     plt.legend(['loss', 'accuracy'], loc='upper left')
        #     plt.show()
        # print('________________________________________________')
        # for single_run in Cartesian_history:
        #     plt.plot(single_run.history['loss'])
        #     plt.plot(single_run.history['accuracy'])
        #     plt.title('Cartesian Run')
        #     plt.xlabel('epoch')
        #     plt.legend(['loss', 'accuracy'], loc='upper left')
        #     plt.show()
        
    if mode == 1:
        K = 5 #if we don't want to train again, run this
        PARAM_BETA_TEST_NUM = 6
    
    # Filematrix
    n = len(image_files)
    m = K * 2

    filematrix = np.zeros((n,m))
    for img_type in ['polar', 'carte']:
        #
        img_extenstion = 'tif'
        #
        for_counter = 0
        if img_type == 'polar':
            working_parent_folder = PARAM_PATH_TEMP_POLAR
        else:
            working_parent_folder = PARAM_PATH_TEMP_CARTE
            for_counter = 1
        for i in range(K):
            image_path = os.path.join(working_parent_folder, str(i), img_type, PARAM_IMG_FOLDER)
            img_pattern = os.path.join(image_path, f'*.{image_extension}')
            image_files = glob.glob(img_pattern)
            for file_name in image_files:
                file_name_shorten = os.path.basename(file_name)
                file_name_raw, ext = os.path.splitext(file_name_shorten)
                filematrix[int(file_name_raw),i + for_counter * K] = 1
            #number_of_ones = np.count_nonzero(filematrix == 1)
            #print(number_of_ones)  #uncomment this line when png file is not satisfactory, we can track the number of ones during each step  
    plt.imsave('filematrix.png', filematrix, cmap = 'binary')   
            
	
