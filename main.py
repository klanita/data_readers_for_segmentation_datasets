# ==================================================================
# import 
# ==================================================================
import load_datasets
import time

# ================================
# set anatomy here
# ================================
# 'brain' / 'cardiac' / 'prostate'
# ================================

# ================================
# set dataset here.
# ================================
# for brain: 'HCP_T1' / 'HCP_T2' / 'ABIDE_caltech' / 'ABIDE_stanford'
# for cardiac: 'ACDC' / 'RVSC'
# for prostate: 'NCI' / 'PIRAD_ERC'
# ================================
# DATASET = 'NCI' 
# DATASET = 'HCP_T1'
# DATASET = 'HCP_T2'
# DATASET = 'ABIDE_caltech'
# DATASET = 'ABIDE_stanford'


# ================================
# set train / test / validation here
# ================================
# 'train' / 'test' / 'validation'
# TRAIN_TEST_VALIDATION = 'train'
# TRAIN_TEST_VALIDATION = 'validation'
# TRAIN_TEST_VALIDATION = 'test'

# ================================
# read images and segmentation labels
# ================================

# ANATOMY = 'brain'
# for TRAIN_TEST_VALIDATION in ['train', 'validation', 'test']:
#     for DATASET in ['HCP_T1', 'HCP_T2', 'ABIDE_caltech', 'ABIDE_stanford']:
#         images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
#                                                     dataset = DATASET,
#                                                     train_test_validation = TRAIN_TEST_VALIDATION,
#                                                     first_run = True)  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)


ANATOMY = 'prostate'
# for DATASET in ['NCI']:
# # for DATASET in ['NCI', 'PIRAD_ERC']:
#     for TRAIN_TEST_VALIDATION in ['train', 'validation', 'test']:
#         ts = time.time()
#         print(ANATOMY, TRAIN_TEST_VALIDATION, DATASET)
#         images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
#                                                     dataset = DATASET,
#                                                     train_test_validation = TRAIN_TEST_VALIDATION,
#                                                     first_run = True)  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)
#         run_time_min = (time.time() - ts)/60
#         print(f'{run_time_min:.1f} min')

DATASET_LIST = {'cardiac': ['ACDC', 'RVSC'],
                'brain': ['HCP_T1', 'HCP_T2', 'ABIDE_caltech', 'ABIDE_stanford'],
                'prostate': ['NCI', ]}
                # 'prostate': ['NCI', 'PIRAD_ERC']}

# ANATOMY = 'cardiac'
# TRAIN_TEST_VALIDATION_LIST = ['train', 'validation', 'test']
TRAIN_TEST_VALIDATION_LIST = ['train']

for DATASET in DATASET_LIST[ANATOMY]:
    for TRAIN_TEST_VALIDATION in TRAIN_TEST_VALIDATION_LIST:
        ts = time.time()
        print(ANATOMY, TRAIN_TEST_VALIDATION, DATASET)
        images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
                                                    dataset = DATASET,
                                                    train_test_validation = TRAIN_TEST_VALIDATION,
                                                    first_run = True)  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)
        run_time_min = (time.time() - ts)/60
        print(len(images), len(labels))
        print(f'{run_time_min:.1f} min')
