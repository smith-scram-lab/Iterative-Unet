from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_float

import sys
rospath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if str(sys.path).find(rospath) != -1:
    sys.path.remove(rospath) # in order to import cv2 under python3
    print('ROS path temporarily removed.')
import cv2


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path, image_folder, mask_folder,
                  image_gray = False, mask_gray = True):
    img_paths = list()
    lbl_paths = list()
    
    # Recursively find all the image files from the path test_path
    for img_path in glob.glob(test_path+"/"+image_folder+"/*"):
        img_paths.append(img_path)
    
    # Recursively find all the image files from the path label_path
    for lbl_path in glob.glob(test_path+"/"+mask_folder+"/*"):
        lbl_paths.append(lbl_path)
        
    images = np.zeros((len(img_paths),256,256,3))
    labels = np.zeros((len(lbl_paths),256,256,1))
      
    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        # Takes as input path to image file and returns 
        # resized 3 channel RGB image of as numpy array of size (256, 256, 3)
        images[i] = np.array(io.imread(img_path, as_gray = image_gray)) / 255
    for i, lbl_path in enumerate(lbl_paths):
        labels[i] = np.array(io.imread(lbl_path, as_gray = mask_gray)).reshape((256,256,1)) / 255

    errmsg1 = 'mismatched dimension: ' + str(len(img_paths))+' images' + str(len(lbl_paths))+' labels'
    errmsg2 = 'no files detected'

    assert len(img_paths) == len(lbl_paths), errmsg1
    assert len(img_paths) > 0, errmsg2

    return images


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    os.mkdir(save_path)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def mergeIm(test_path, img_folder, mask_folder, results_path, save_path):
    list = os.listdir(os.path.join(test_path,img_folder))
    number_files = len(list)
    
    # Paths for test, result, label images
    path_test = os.path.join(os.path.join(test_path,img_folder), "*.tif")
    path_label = os.path.join(os.path.join(test_path, mask_folder), "*.tif")
    path_result = os.path.join(results_path, "*.png")
    
    
    # Information of images
    images_test = [cv2.imread(img) for img in glob.glob(path_test)]
    images_result = [cv2.imread(img) for img in glob.glob(path_result)]
    images_label = [cv2.imread(img) for img in glob.glob(path_label)]
    
    h,w,d = images_test[0].shape
    
    height = h * number_files
    width = w * 3
    output = np.zeros((height,width,3))
    
    # current row
    n = 0
    for i in range(number_files):
        # test image | result image | ground truth
        output[n:n+h,0:w] = images_test[i]
        output[n:n+h,w:w*2] = images_result[i]
        output[n:n+h,w*2:w*3] = images_label[i]
        n += h
    
    cv2.imwrite(os.path.join(save_path), output)
