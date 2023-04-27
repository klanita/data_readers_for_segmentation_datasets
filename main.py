# ==================================================================
# import 
# ==================================================================
import load_datasets

# ================================
# set anatomy here
# ================================
# 'brain' / 'cardiac' / 'prostate'
# ================================
# ANATOMY = 'prostate'
ANATOMY = 'brain'

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
TRAIN_TEST_VALIDATION = 'test'

# ================================
# read images and segmentation labels
# ================================

for TRAIN_TEST_VALIDATION in ['train', 'validation', 'test']:
    for DATASET in ['HCP_T1', 'HCP_T2', 'ABIDE_caltech', 'ABIDE_stanford']:
        images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
                                                    dataset = DATASET,
                                                    train_test_validation = TRAIN_TEST_VALIDATION,
                                                    first_run = True)  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)

