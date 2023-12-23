import numpy as np 
import os 
from data import *
from model import *
from defines import *
import math


K = 5
batch_size = 5

cartesian_original_folder = 'data/endoscopic/cartesian'
polar_original_folder = 'data/endoscopic/polar'
cartesian_test_folder = './data/endoscopic_test956/cartesian'
polar_test_folder = './data/endoscopic_test956/polar'
cartesian_result_file = './big_carte_prediction.npy'
polar_result_file = './big_polar_prediction.npy'

gen_args = [CARTE_GEN_ARGS,POLAR_GEN_ARGS]
folders = [cartesian_original_folder,polar_original_folder]
test_folders = [cartesian_test_folder,polar_test_folder]
result_files = [cartesian_result_file, polar_result_file]

for i in range(2):
    train_gene = trainGenerator(batch_size, folders[i], PARAM_IMG_FOLDER, PARAM_MSK_FOLDER, gen_args[i])
    model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM])
    model_checkpoint_file = './big_model.hdf5'
    model_checkpoint = ModelCheckpoint(model_checkpoint_file,monitor = 'loss', verbose = 1, save_best_only=True)
    force_restart_cumulative_count = 0
    force_restart_count = 0
    previou_min_loss = math.inf
    keepGoing = False
    while(keepGoing):
        test_run = model.fit(train_gene, verbose = 1, steps_per_epoch = STEPS, epochs = EPOCHS, callbacks = [model_checkpoint])
        force_restart_cumulative_count += EPOCHS
        current_min = min(test_run.history['loss'])
        if current_min <= previou_min_loss:
            previou_min_loss = current_min
            force_restart_count = 0                
        else:
            if previou_min_loss < TRAIN_STOP_THRESHOLD: 
                keepGoing = False
            else:
                if force_restart_count >= FORCE_RESTART_TOLERANCE and force_restart_cumulative_count >= CUMULATIVE_STOP_TOLERANCE:
                    force_restart_count = 0
                    force_restart_cumulative_count = 0
                    previou_min_loss = math.inf
                    os.remove(model_checkpoint_file)
                    model_checkpoint = ModelCheckpoint(model_checkpoint_file, monitor = 'loss', verbose=1, save_best_only=True)
                    model = unet(PARAM_BETA1[PARAM_BETA_TEST_NUM], PARAM_BETA2[PARAM_BETA_TEST_NUM]) 
                else:
                    force_restart_count += 1
                    model.load_weights(model_checkpoint_file)


    test_gene = testGenerator(test_folders[i], PARAM_IMG_FOLDER, PARAM_MSK_FOLDER)
    test_results = model.predict_generator(test_gene, 956, verbose=1)
    np.save(result_files[i], test_results)
