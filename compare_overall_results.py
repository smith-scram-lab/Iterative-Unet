import numpy as np
import os
import cv2
from p2ctransformer.p2c import *
polar_prediction_file = 'big_polar_prediction.npy'
carte_prediction_file = 'big_carte_prediction.npy'
polar_prediction = np.load(polar_prediction_file)
carte_prediction = np.load(carte_prediction_file)
test_label_folder_path = 'data/endoscopic_test956/cartesian/label'
count = 0
big_polar_dice = []
big_carte_dice = []
for prediction in [polar_prediction,carte_prediction]:
    for i in range(956):
        test_file_name = os.path.join(test_label_folder_path, (str(i) + '.tif'))
        ground_truth_mask = cv2.imread(test_file_name, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = ground_truth_mask / 255.0
        ground_truth_mask = ground_truth_mask.astype(np.uint8)
        current_prediction = prediction[i]
        current_prediction = np.reshape(current_prediction,(256,256))
        threshold = 0.5
        current_prediction = (current_prediction > threshold).astype(np.uint8)
        if count == 0:
            current_prediction = p2c(current_prediction, trans_dic)
            big_polar_dice.append(dice_coefficient_carte(ground_truth_mask,current_prediction))
        else:
            big_carte_dice.append(dice_coefficient_carte(ground_truth_mask,current_prediction))        
    count += 1
big_polar_dice = np.asarray(big_polar_dice)
big_carte_dice = np.asarray(big_carte_dice)

np.save('big_carte_prediction_score.npy', big_carte_dice)
np.save('big_polar_prediction_score.npy', big_polar_dice)