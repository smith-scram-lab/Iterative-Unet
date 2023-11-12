from defines import *
from model import *
from data import *
from filePrep import *
#from migration_yz.migrator import *
from migration_cl.migrator import *
#from migration_cw.migrator import *
from model_reader.modelreader import *
from p2ctransformer.p2c import *

import cv2

from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import random
import multiprocessing

PARAM_BETA_TEST_NUM = 6
K = 5
batch_size = 1

def train(working_parent_folder,data_gen_args, queue):    
    history = []
    if_polar = False
    if working_parent_folder == PARAM_PATH_TEMP_POLAR:
        if_polar = True
    
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
        print('Now Training the Model for folder',i)
        if if_polar:
            print('Now in polar group')
        else:
            print('Now in Cartesian group')
        test_run = model.fit(test_gene, verbose = 1, steps_per_epoch = STEPS, epochs = EPOCHS, callbacks = [model_checkpoint])
        history.append(test_run)
        shutil.rmtree(temp_folder_path)
        loss_curve = []
        for eachrun in history:
            loss_curve.append(eachrun.history['loss'])
    queue.put(loss_curve) 

def test(filematrix, queue):
    n = filematrix.shape[0]
    m = K * 2
    scorematrix = np.zeros((n,m))
    image_extension = 'tif'
    augmented_filematrix = np.copy(filematrix)
    for row in augmented_filematrix:
        for for_counter in range(2):
            zero_count = 0
            for index in range(K):
                real_index = index + for_counter * K
                if row[real_index] == 0:
                    zero_count += 1
            if zero_count == 5:
                row[for_counter*K:for_counter*K + 5] = 1
    row_indices, col_indices = np.where(augmented_filematrix == 1)
    indices = list(zip(row_indices, col_indices))
    scorematrix = np.zeros((n,m))
    for img_type in ['polar', 'carte']:
        for_counter = 0
        if img_type == 'polar':
            working_parent_folder = PARAM_PATH_TEMP_POLAR
            src_folder = PARAM_PATH_POLAR
        else:
            working_parent_folder = PARAM_PATH_TEMP_CARTE
            src_folder = PARAM_PATH_CARTE
            for_counter = 1
            
        for i in range(K):
            current_folder_index = i + for_counter * K
            temp_test_folder_name = 'temptest'
            #print(working_parent_folder)
                
            if os.path.exists(temp_test_folder_name):
                shutil.rmtree(temp_test_folder_name)
            temp_test_img_folder = os.path.join(temp_test_folder_name,PARAM_IMG_FOLDER)
            temp_test_msk_folder = os.path.join(temp_test_folder_name,PARAM_MSK_FOLDER)
            os.makedirs(temp_test_img_folder)
            os.makedirs(temp_test_msk_folder)
        
            for indice in indices:
                if indice[1] == current_folder_index:
                    img_name = str(indice[0]) + '.' + image_extension
                    src = os.path.join(src_folder,PARAM_IMG_FOLDER,img_name)
                    shutil.copy2(src, temp_test_img_folder)
                    src = os.path.join(src_folder,PARAM_MSK_FOLDER,img_name)
                    shutil.copy2(src, temp_test_msk_folder)
            model_path = os.path.join(working_parent_folder, str(i), 'checkpoint.hdf5')
            print('Now working with path', model_path)
            current_model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
            current_model.load_weights(model_path) 
            for test_image_name in os.listdir(temp_test_img_folder):
                test_image_name_raw, ext = os.path.splitext(test_image_name)
                image_path = os.path.join(temp_test_img_folder, test_image_name)
                ground_truth_mask_path = os.path.join(temp_test_msk_folder, test_image_name)
                
                test_image = cv2.imread(image_path)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                test_image = test_image / 255.0
                test_image = np.expand_dims(test_image,axis = 0)

                ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
                ground_truth_mask = ground_truth_mask / 255.0
                ground_truth_mask = ground_truth_mask.astype(np.uint8)
                ###HERE IS WHERE PREDICT AND GENERATE SCORE
                prediction = current_model.predict(test_image, verbose = 0)
                
                threshold = 0.5
                binary_mask = (prediction > threshold).astype(np.uint8)
                binary_mask = binary_mask[0,:,:,0]
                if img_type == 'polar':
                    dice = dice_coefficient(ground_truth_mask, binary_mask)
                else:
                    dice = dice_coefficient_carte(ground_truth_mask, binary_mask)
                ###REPLACE WITH QUICK SCORE GENERATOR TO DEBUG THE ITERATION GROUP
                ###HERE IS THE QUICK SCORE GENERATOR 
                #dice = random.random()
                ###REPLACE WITH REAL PREDICT BLOCK FOR NORMAL ACTION
                scorematrix[int(test_image_name_raw), current_folder_index] = dice   
    queue.put(scorematrix)
         
def test_allinc(filematrix, queue):
    n = filematrix.shape[0]
    m = K * 2
    scorematrix = np.zeros((n,m))
    trans_dic = p2c_dic_gen(127, 127, 256, 256)
    image_extension = 'tif'
    augmented_filematrix = np.copy(filematrix)
    for row in augmented_filematrix:
        for for_counter in range(2):
            zero_count = 0
            for index in range(K):
                real_index = index + for_counter * K
                if row[real_index] == 0:
                    zero_count += 1
            if zero_count == 5:
                row[for_counter*K:for_counter*K + 5] = 1
    row_indices, col_indices = np.where(augmented_filematrix == 1)
    indices = list(zip(row_indices, col_indices))
    scorematrix = np.zeros((n,m))
    for img_type in ['polar', 'carte']:
        for_counter = 0
        ground_truth_src_folder = PARAM_PATH_CARTE
        if img_type == 'polar':
            working_parent_folder = PARAM_PATH_TEMP_POLAR
            src_folder = PARAM_PATH_POLAR
        else:
            working_parent_folder = PARAM_PATH_TEMP_CARTE
            src_folder = PARAM_PATH_CARTE
            for_counter = 1
            
        for i in range(K):
            current_folder_index = i + for_counter * K
            temp_test_folder_name = 'temptest'
            #print(working_parent_folder)
                
            if os.path.exists(temp_test_folder_name):
                shutil.rmtree(temp_test_folder_name)
            temp_test_img_folder = os.path.join(temp_test_folder_name,PARAM_IMG_FOLDER)
            temp_test_msk_folder = os.path.join(temp_test_folder_name,PARAM_MSK_FOLDER)
            os.makedirs(temp_test_img_folder)
            os.makedirs(temp_test_msk_folder)
        
            for indice in indices:
                if indice[1] == current_folder_index:
                    img_name = str(indice[0]) + '.' + image_extension
                    src = os.path.join(src_folder,PARAM_IMG_FOLDER,img_name)
                    shutil.copy2(src, temp_test_img_folder)
                    src = os.path.join(ground_truth_src_folder,PARAM_MSK_FOLDER,img_name)
                    shutil.copy2(src, temp_test_msk_folder)
            model_path = os.path.join(working_parent_folder, str(i), 'checkpoint.hdf5')
            print('Now working with path', model_path)
            current_model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
            current_model.load_weights(model_path) 
            for test_image_name in os.listdir(temp_test_img_folder):
                test_image_name_raw, ext = os.path.splitext(test_image_name)
                image_path = os.path.join(temp_test_img_folder, test_image_name)
                ground_truth_mask_path = os.path.join(temp_test_msk_folder, test_image_name)
                
                test_image = cv2.imread(image_path)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                test_image = test_image / 255.0
                test_image = np.expand_dims(test_image,axis = 0)

                ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
                ground_truth_mask = ground_truth_mask / 255.0
                ground_truth_mask = ground_truth_mask.astype(np.uint8)
                ###HERE IS WHERE PREDICT AND GENERATE SCORE
                '''prediction = current_model.predict(test_image, verbose = 0)
                threshold = 0.5
                binary_mask = (prediction > threshold).astype(np.uint8)
                binary_mask = binary_mask[0,:,:,0]
                if img_type == 'polar':
                    binary_mask = p2c(binary_mask, trans_dic)
                dice = dice_coefficient_carte(ground_truth_mask, binary_mask)'''
                ###REPLACE WITH QUICK SCORE GENERATOR TO DEBUG THE ITERATION GROUP
                ###HERE IS THE QUICK SCORE GENERATOR 
                dice = random.random()
                ###REPLACE WITH REAL PREDICT BLOCK FOR NORMAL ACTION
                scorematrix[int(test_image_name_raw), current_folder_index] = dice   
    queue.put(scorematrix)

def train_and_test_last_round(migrating_wizard):
    split = migrating_wizard.get_loc_current()
    
    true_indices = np.where(split)[0]
    false_indices = np.where(~split)[0]
    if os.path.exists('./temp_lastround'):
        shutil.rmtree('./temp_lastround')
    os.makedirs('./temp_lastround/cartesian_Dom/image')
    os.makedirs('./temp_lastround/cartesian_Dom/label')
    os.makedirs('./temp_lastround/polar_Dom/image')
    os.makedirs('./temp_lastround/polar_Dom/label')
    
    i = 0
    for item in true_indices:
        img_name = str(item) + '.tif'
        src = os.path.join(PARAM_PATH_POLAR, PARAM_IMG_FOLDER, img_name)
        shutil.copy2(src, os.path.join('./temp_lastround/polar_Dom/image'))
        src = os.path.join(PARAM_PATH_POLAR, PARAM_MSK_FOLDER, img_name)
        shutil.copy2(src, os.path.join('./temp_lastround/polar_Dom/label'))
    for item in false_indices:
        img_name = str(item) + '.tif'
        src = os.path.join(PARAM_PATH_CARTE, PARAM_IMG_FOLDER, img_name)
        shutil.copy2(src, os.path.join('./temp_lastround/cartesian_Dom/image'))
        src = os.path.join(PARAM_PATH_CARTE, PARAM_MSK_FOLDER, img_name)
        shutil.copy2(src, os.path.join('./temp_lastround/cartesian_Dom/label'))
    polar_data_gen_args = dict(rotation_range = 50,      # TODO: improve the data augmentation
                width_shift_range =0.2,
                height_shift_range =0.2,
                shear_range = 0.35,
                zoom_range = 0.05,
                horizontal_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
    polar_train_gene = trainGenerator(batch_size, './temp_lastround/polar_Dom', PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, polar_data_gen_args)
    polar_model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
    polar_model_checkpoint = ModelCheckpoint('./temp_lastround/polar_Dom/checkpoint.hdf5', monitor = 'loss', verbose=1, save_best_only=True)
    polar_model.fit(polar_train_gene, verbose = 1, steps_per_epoch = STEPS, epochs = EPOCHS, callbacks = [polar_model_checkpoint])
    polar_test_gene = testGenerator('./data/endoscopic_test956/polar', PARAM_IMG_FOLDER, PARAM_MSK_FOLDER)
    polar_results = polar_model.predict_generator(polar_test_gene, 956, verbose=1)
    np.save('./results/polar_prediction.npy',polar_results)

    carte_data_gen_args = dict(rotation_range = 80,      # TODO: improve the data augmentation
                width_shift_range =0.02,
                height_shift_range =0.02,
                shear_range = 0.35,
                zoom_range = 0.075,
                horizontal_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
    carte_train_gene = trainGenerator(batch_size, './temp_lastround/cartesian_Dom', PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, carte_data_gen_args)
    carte_model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
    carte_model_checkpoint = ModelCheckpoint('./temp_lastround/cartesian_Dom/checkpoint.hdf5', monitor = 'loss', verbose=1, save_best_only=True)
    carte_model.fit(carte_train_gene, verbose = 1, steps_per_epoch = STEPS, epochs = EPOCHS, callbacks = [carte_model_checkpoint])
    carte_test_gene = testGenerator('./data/endoscopic_test956/cartesian', PARAM_IMG_FOLDER, PARAM_MSK_FOLDER)
    carte_results = carte_model.predict_generator(carte_test_gene, 956, verbose=1)
    np.save('./results/carte_prediction.npy',carte_results)

def train_2K_models(round):        
    queue = multiprocessing.Queue()
    data_gen_args = dict(rotation_range = 50,      # TODO: improve the data augmentation
                width_shift_range =0.2,
                height_shift_range =0.2,
                shear_range = 0.35,
                zoom_range = 0.05,
                horizontal_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
    PP = multiprocessing.Process(target=train, args= ([PARAM_PATH_TEMP_POLAR, data_gen_args, queue]))
    PP.start()
    polar_history = queue.get()
    PP.join()
    
    data_gen_args = dict(rotation_range = 80,      # TODO: improve the data augmentation
                width_shift_range =0.02,
                height_shift_range =0.02,
                shear_range = 0.35,
                zoom_range = 0.075,
                horizontal_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
    PC = multiprocessing.Process(target=train, args= ([PARAM_PATH_TEMP_CARTE, data_gen_args, queue]))
    PC.start()
    carte_history = queue.get()
    PC.join()
    #model_PNGgen(polar_history, carte_history, round)
    #print(polar_history)
    model_npyStore(polar_history,carte_history,round)
    
def model_npyStore(polar_history, carte_history, round):
    models_history_path = os.path.join(PARAM_RESULTS,'models_history/round_'+str(round))
    os.makedirs(models_history_path, exist_ok = True)
    np_p_history = np.asarray(polar_history)
    np_c_history = np.asarray(carte_history)
    np_p_history_path = os.path.join(models_history_path, 'polar_history.npy')
    np_c_history_path = os.path.join(models_history_path, 'carte_history.npy')
    np.save(np_p_history_path,np_p_history)
    np.save(np_c_history_path,np_c_history)
    


def model_PNGgen(polar_history,carte_history,round):
    models_path = os.path.join(PARAM_RESULTS,'models/round_'+str(round))
    os.makedirs(models_path, exist_ok = True)
    i=0
    for single_run in polar_history:
        plt.plot(single_run.history['loss'])
        plt.plot(single_run.history['accuracy'])
        plt.title('Polar Run')
        plt.xlabel('epoch')
        plt.legend(['loss', 'accuracy'], loc='upper left')
        plt.savefig(os.path.join(models_path,'polar_'+str(i)+'.jpg'))
        i+=1

    i =0
    for single_run in carte_history:
        plt.plot(single_run.history['loss'])
        plt.plot(single_run.history['accuracy'])
        plt.title('Cartesian Run')
        plt.xlabel('epoch')
        plt.legend(['loss', 'accuracy'], loc='upper left')
        plt.savefig(os.path.join(models_path,'carte_'+str(i)+'.jpg'))    
        i+=1
    print("models saved")

def dice_coefficient(image1, image2):#Generate the Dice coefficient of two binary images, should do thresholding before inputting
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
   
    # Calculate the Dice coefficient
    dice = (2.0 * intersection_o + smooth) / (sum_image1_o + sum_image2_o + smooth)
    #print('dice',dice)
    dice_f = (2.0 * intersection_f + smooth) / (sum_image1_f + sum_image2_f + smooth)
    #print('dice_f',dice_f)
    dice_avg = (dice + dice_f) / 2.0
    #print('dice_avg', dice_avg)
    return dice_avg

def dice_coefficient_carte(image1, image2):#Generate the Dice coefficient of two binary images, should do thresholding before inputting
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
   
    # Calculate the Dice coefficient
    dice = (2.0 * intersection_o + smooth) / (sum_image1_o + sum_image2_o + smooth)
    #print('dice',dice)
    dice_f = (2.0 * (intersection_f - 14616) + smooth) / (sum_image1_f + sum_image2_f + smooth - 29232) #Hard-coded numbers here, need to prove 
    #print('dice_f',dice_f)
    dice_avg = (dice + dice_f) / 2.0
    #print('dice_avg', dice_avg)
    return dice_avg

def make_K_folds(polar_indices,carte_indices,K):
    checkNcreateTempFolder(PARAM_PATH_TEMP_POLAR, K)
    checkNcreateTempFolder(PARAM_PATH_TEMP_CARTE, K)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    i = 0
    for train_indices, test_indices_P in kfold.split(polar_indices):
        fillFolder(test_indices_P, polar_indices, PARAM_PATH_POLAR, PARAM_PATH_CARTE, PARAM_PATH_TEMP_POLAR, i)
        print('Polar temp folder', i, 'created')
        i += 1
    i = 0
    for train_indices, test_indices_C in kfold.split(carte_indices):
        fillFolder(test_indices_C, carte_indices, PARAM_PATH_POLAR, PARAM_PATH_CARTE, PARAM_PATH_TEMP_CARTE, i)
        print('Cartesian temp folder', i, 'created')
        i += 1
    return filematrixPNG_gen(K)


def filematrixPNG_gen(K):
    image_extension = 'tif'
    img_pattern = os.path.join(PARAM_PATH_POLAR, PARAM_IMG_FOLDER, f'*.{image_extension}')
    image_files = glob.glob(img_pattern)
    n = len(image_files)
    m = K * 2
    filematrix = np.zeros((n,m))
    for img_type in ['polar', 'carte']:
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
    
    # filematrix_name = 'filematrix/filematrix_round_'+str(round)+'.png'
    # filematrix_path = os.path.join(PARAM_RESULTS,filematrix_name)
    # plt.imsave(filematrix_path, file_matrix, cmap = 'binary')
    print('File Location saved as filematrix.png')
    return filematrix
    

######_________________________DEBUGGING TOOL______________

######_________________________MAIN________________________
if __name__ == '__main__':
    # step0: enable GPU version
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.system("tree -d")
    mov_count_his = []
    is_first_round = True
    K = 5
    model_reader = modelreader()
    #while True:   
    for round in range(1):
    # step1: file relocation 
        print('Now is round', round)
        if is_first_round:
            is_first_round = False
            score_file_polar = 'analysis_dice_back_Train_P.npy'
            score_file_carte = 'analysis_dice_back_Train_C.npy'
            np_file_polar = os.path.join(PARAM_PATH_SCORES, score_file_polar)
            np_file_carte = os.path.join(PARAM_PATH_SCORES, score_file_carte)
            img_score_polar = np.load(np_file_polar)
            img_score_carte = np.load(np_file_carte)
            migrating_wizard = migrator(img_score_polar,img_score_carte, K, ifFlip = True)
            first_split = migrating_wizard.get_loc_current()
            true_indices = np.where(first_split)[0]
            false_indices = np.where(~first_split)[0]

            file_matrix = make_K_folds(true_indices,false_indices,K)
            
            
        else:
            split = migrating_wizard.get_loc_current()
            true_indices = np.where(split)[0]
            false_indices = np.where(~split)[0]
            file_matrix = make_K_folds(true_indices,false_indices,K)

        filematrix_name = 'filematrix/filematrix_round_'+str(round)+'.png'
        filematrix_path = os.path.join(PARAM_RESULTS,filematrix_name)
        plt.imsave(filematrix_path, file_matrix, cmap = 'binary')

        #now that we have all the temporary folders ready, we train the ten models
        train_2K_models(round)

        #model_PNGgen(polar_history,carte_history,round)
        queue = multiprocessing.Queue()
        PT = multiprocessing.Process(target=test_allinc,args=([file_matrix,queue]))
        PT.start()
        scorematrix = queue.get()
        PT.join()
        scorematrix_name = 'scorematrix/scorematrix_round_' + str(round) + '.npy'
        scorematrix_path = os.path.join(PARAM_RESULTS,scorematrix_name)
        np.save(scorematrix_path, scorematrix)

        #this is yz method
        '''migrating_wizard.decide_and_mod_prob(scorematrix)
        migrating_wizard.migrate()'''
        #End of yz method

        #starting here is cl method
        dif, decision = migrating_wizard.get_decision(K, scorematrix)
        count_p2c_c2p = migrating_wizard.decide_move(2000, dif, decision)
        mov_count_his.append(count_p2c_c2p)
        #End of cl method
        
        #Start of cw method
        #migrating_wizard.migrate(scorematrix)
        #End of cw method

        history = migrating_wizard.get_loc_history()
        history_name = 'history/history_round_' + str(round) + '.npy'
        history_path = os.path.join(PARAM_RESULTS,history_name)
        np.save(history_path, history)
    print(mov_count_his)
    train_and_test_last_round(migrating_wizard)
        #uncomment if yz method
    '''prob_history = migrating_wizard.get_prob_history() 
    prob_history_name = 'prob_history/prob_history_round_' + str(round) + '.npy'
    prob_history_path = os.path.join(PARAM_RESULTS,prob_history_name)
    np.save(prob_history_path, prob_history)'''
        


          
'''for job in jobs:
    job.join()
	# if 1st round:
        #First round should have the two scorefiles input into the migrator, generate the first round of location matrix 
        #sort scores in descending order and store index
        #sorted only on cartesian score. tbc....
        #fetch top polar dominant and non-polar dominant image

    # else:
        # load scores from last round:

        # relocate files
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

    #step 2: generate new training model
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
        
        # Generate Plot loss
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
    
        # Filematrix 7404*10 [0000000100]
        n = len(image_files)  #7404
        m = K * 2             #10migrating_wizard.migrate()  
            print(migrating_wizard.get_loc_history()[0:10])
        img_extenstion = 'tif'
        filematrix = np.zeros((n,m))
        for img_type in ['polar', 'carte']:

            if_carte = 0
            if img_type == 'polar':
                working_parent_folder = PARAM_PATH_TEMP_POLAR
            else:
                working_parent_folder = PARAM_PATH_TEMP_CARTE
                if_carte = 1
            for i in range(K):
                image_path = os.path.join(working_parent_folder, str(i), img_type, PARAM_IMG_FOLDER)
                img_pattern = os.path.join(image_path, f'*.{image_extension}')
                image_files = glob.glob(img_pattern)
                for file_name in image_files:
                    file_name_shorten = os.path.basename(file_name)
                    file_name_raw, ext = os.path.splitext(file_name_shorten)
                    filematrix[int(file_name_raw), i + if_carte * K] = 1
                #number_of_ones = np.count_nonzero(filematrix == 1)
                #print(number_of_ones)  #uncomment this line when png file is not satisfactory, we can track the number of ones during each step  
        plt.imsave('filematrix.png', filematrix, cmap = 'binary')   

        # Augmented Filematrix 7404*10 [1111100100]
        augmented_filematrix = np.copy(filematrix)
        for row in augmented_filematrix:
            
            for if_carte in range(2):
                zero_count = 0
                for index in range(K):
                    real_index = index + if_carte * K
                    if row[real_index] == 0:
                        zero_count += 1
                if zero_count == 5:
                    row[if_carte*K:if_carte*K + 5] = 1
        
        row_indices, col_indices = np.where(augmented_filematrix == 1)
        indices = list(zip(row_indices, col_indices))


        # generate scorematrix with test results [000a0bcdef]
        scorematrix = np.zeros((n,m))
        for img_type in ['polar', 'carte']:
            if_carte = 0

            if img_type == 'polar':
                working_parent_folder = PARAM_PATH_TEMP_POLAR
                src_folder = PARAM_PATH_POLAR
            else:
                working_parent_folder = PARAM_PATH_TEMP_CARTE
                src_folder = PARAM_PATH_CARTE
                if_carte = 1
                
            for i in range(K):
                current_folder_index = i + if_carte * K
                temp_test_folder_name = 'temptest'
                #print(working_parent_folder)
                    
                if os.path.exists(temp_test_folder_name):
                    shutil.rmtree(temp_test_folder_name)
                temp_test_img_folder = os.path.join(temp_test_folder_name,PARAM_IMG_FOLDER)
                temp_test_msk_folder = os.path.join(temp_test_folder_name,PARAM_MSK_FOLDER)
                os.makedirs(temp_test_img_folder)
                os.makedirs(temp_test_msk_folder)
            
                for indice in indices:
                    if indice[1] == current_folder_index:
                        img_name = str(indice[0]) + '.' + img_extenstion
                        src = os.path.join(src_folder,PARAM_IMG_FOLDER,img_name)
                        shutil.copy2(src, temp_test_img_folder)
                        src = os.path.join(src_folder,PARAM_MSK_FOLDER,img_name)
                        shutil.copy2(src, temp_test_msk_folder)
                
                model_path = os.path.join(working_parent_folder, str(i), 'checkpoint.hdf5')
                current_model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
                current_model.load_weights(model_path) 
                for test_image_name in os.listdir(temp_test_img_folder):
                    test_image_name_raw, ext = os.path.splitext(test_image_name)
                    image_path = os.path.join(temp_test_img_folder, test_image_name)
                    ground_truth_mask_path = os.path.join(temp_test_msk_folder, test_image_name)
                    
                    test_image = cv2.imread(image_path)
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                    test_image = test_image / 255.0
                    test_image = np.expand_dims(test_image,axis = 0)

                    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
                    ground_truth_mask = ground_truth_mask / 255.0
                    ground_truth_mask = ground_truth_mask.astype(np.uint8)
                    
                    prediction = current_model.predict(test_image, verbose = 0)
                    
                    threshold = 0.5
                    binary_mask = (prediction > threshold).astype(np.uint8)
                    binary_mask = binary_mask[0,:,:,0]
                    dice = dice_coefficient(ground_truth_mask, binary_mask)
                    scorematrix[int(test_image_name_raw), current_folder_index] = dice
                print('Done with folder ', current_folder_index)

                #save scorematrix'''