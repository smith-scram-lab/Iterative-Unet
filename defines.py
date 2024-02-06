from datetime import datetime

PARAM_SYSTEM_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
PARAM_PATH_CARTE = './data/endoscopic/cartesian'
PARAM_PATH_POLAR = './data/endoscopic/polar'
PARAM_PATH_SCORES = './prev-analysis'
PARAM_PATH_TEMP_CARTE = './temp/cartesian_Dom'
PARAM_PATH_TEMP_POLAR = './temp/polar_Dom'
PARMA_PATH_TEST_CARTE = './data/endoscopic_test956/cartesian'
PARAM_PATH_TEST_POLAR = './data/endoscopic_test956/polar'
PARAM_PATH_TEMP_POLAR = './temp/polar_Dom'
PARMA_PATH_TEST_CARTE = './data/endoscopic_test956/cartesian'
PARAM_PATH_TEST_POLAR = './data/endoscopic_test956/polar'

PARAM_BETA1 = [0.2, 0.4, 0.6, 0.4, 0.2, 0.5, 0.5]
PARAM_BETA2 = [0.8, 0.4, 0.8, 0.999, 0.999, 0.5, 0.999]

PARAM_SUB_FOLDER_POLAR = 'polar'
PARAM_SUB_FOLDER_CARTE = 'cartesian'
PARAM_IMG_FOLDER = 'image'
PARAM_MSK_FOLDER = 'label'

PARAM_RESULTS = 'results'
PARAM_SPLIT_NUM = 5

PARAM_BETA_TEST_NUM = 6

STEPS = 100
EPOCHS = 5
ROUND = 15
FORCE_RESTART_TOLERANCE = 3
CUMULATIVE_STOP_TOLERANCE = 50
TRAIN_STOP_THRESHOLD = 0.25

RANDOM_SEED = 43

TARGET_HEIGHT = 256
TARGET_WIDTH = 256
CHANNEL_NUM = 3

POLAR_GEN_ARGS = dict(horizontal_flip = True,
                vertical_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
CARTE_GEN_ARGS = dict(rotation_range = 360,      # TODO: improve the data augmentation
                horizontal_flip = True,
                vertical_flip = True,
                fill_mode = 'nearest',
                rescale = 1./255)
