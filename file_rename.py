import os
import numpy as np
import shutil
from defines import *

def copy_and_rename_folders(original_folders, new_folders,index_to_be_picked):
    files = os.listdir(original_folders[0])
    count = 0
    for index, file_name in enumerate(files):
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        if not index_to_be_picked[int(base_name)]:
            #print(file_name)
            for original_folder, new_folder in zip(original_folders,new_folders):
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                original_path = os.path.join(original_folder, file_name)
                new_file_name = str(count) + '.tif'
                new_path = os.path.join(new_folder,new_file_name)
                shutil.copy2(original_path,new_path)
            count += 1
    

if __name__ == "__main__":
    new_parent_path_name = 'picked_data'
    original_polar_image_path = os.path.join(PARAM_PATH_POLAR,PARAM_IMG_FOLDER)
    original_polar_label_path = os.path.join(PARAM_PATH_POLAR,PARAM_MSK_FOLDER)
    original_carte_image_path = os.path.join(PARAM_PATH_CARTE,PARAM_IMG_FOLDER)
    original_carte_label_path = os.path.join(PARAM_PATH_CARTE,PARAM_MSK_FOLDER)
    new_polar_image_path = os.path.join(new_parent_path_name,original_polar_image_path)
    new_polar_label_path = os.path.join(new_parent_path_name,original_polar_label_path)
    new_carte_image_path = os.path.join(new_parent_path_name,original_carte_image_path)
    new_carte_label_path = os.path.join(new_parent_path_name,original_carte_label_path)
    original_folders = [original_polar_image_path, original_polar_label_path, original_carte_image_path, original_carte_label_path]
    new_folders = [new_polar_image_path,new_polar_label_path,new_carte_image_path,new_carte_label_path]
    picked_index = np.load('picked_index.npy')
    print(picked_index)
    copy_and_rename_folders(original_folders, new_folders,picked_index)