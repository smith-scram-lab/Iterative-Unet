import numpy as np
import os
import cv2
from p2ctransformer.p2c import *

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

polar_prediction_file = 'big_polar_prediction.npy'
carte_prediction_file = 'big_carte_prediction.npy'
polar_prediction_new_file = 'results/polar_prediction.npy'
carte_prediction_new_file = 'results/carte_prediction.npy'
polar_prediction = np.load(polar_prediction_file)
carte_prediction = np.load(carte_prediction_file)
polar_prediction_new = np.load(polar_prediction_new_file)
carte_prediction_new = np.load(carte_prediction_new_file)
test_label_folder_path = 'data/endoscopic_test956/cartesian/label'
file_name = ['polar_overall.npy','carte_overall.npy','polar_new_pre.npy','carte_new_pre.npy']
count = 0
overall_dice = []
trans_dic = p2c_dic_gen(127, 127, 256, 256)
predictions = [polar_prediction, carte_prediction, polar_prediction_new, carte_prediction_new]
for j in range(4):
    prediction = predictions[j]
    for i in range(956):
        test_file_name = os.path.join(test_label_folder_path, (str(i) + '.tif'))
        ground_truth_mask = cv2.imread(test_file_name, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = ground_truth_mask / 255.0
        ground_truth_mask = ground_truth_mask.astype(np.uint8)
        current_prediction = prediction[i]
        current_prediction = np.reshape(current_prediction,(256,256))
        threshold = 0.5
        current_prediction = (current_prediction > threshold).astype(np.uint8)
        if count%2 == 0:
            print('polar', count)
            current_prediction = p2c(current_prediction, trans_dic)
        overall_dice.append(dice_coefficient_carte(ground_truth_mask,current_prediction))   
        overall_dice_np = np.asarray(overall_dice)   
        np.save(file_name[j], overall_dice_np)
    count += 1
