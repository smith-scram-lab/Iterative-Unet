from migration_yz.migrator import *
import numpy as np
import os
from defines import *
score_file_polar = 'analysis_dice_back_Train_P.npy'
score_file_carte = 'analysis_dice_back_Train_C.npy'
np_file_polar = os.path.join(PARAM_PATH_SCORES, score_file_polar)
np_file_carte = os.path.join(PARAM_PATH_SCORES, score_file_carte)
img_score_polar = np.load(np_file_polar)
img_score_carte = np.load(np_file_carte)
migrating_wizard = migrator(img_score_polar,img_score_carte, 5)
print(migrating_wizard.get_loc_history().shape)
score_matrix = np.load('scorematrix/scorematrix_round_0.npy')
print(score_matrix.shape)
migrating_wizard.decide_and_mod_prob(score_matrix)
migrating_wizard.migrate()
np.save('current_prob.npy',migrating_wizard.get_prob_current())

