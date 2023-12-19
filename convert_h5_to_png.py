from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pprint
import os
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from PIL import Image
import h5py
import os
import numpy as np


DATASET_PATHS = {
    'hcp1': {
        'train': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5',
        'val': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp1_full': {
        'train': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_1040.hdf5',
        'val': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        # 'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp2_full': {
        'train': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_1040.hdf5',
        'val': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        # 'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp2': {
        'train': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5',
        'val': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        'test': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'abide_caltech': {
        'train': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_10.hdf5',
        'val': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_10_to_15.hdf5',
        'test': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_16_to_36.hdf5'
        },
    'abide_stanford': {
        'train': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_0_to_10.hdf5',
        'val': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_10_to_15.hdf5',
        'test': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_16_to_36.hdf5'
        },
    'nci': {
        'all': 'nci/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5',
        },
    'pirad_erc': {
        'train': 'pirad_erc/data_2d_from_40_to_68_size_256_256_res_0.625_0.625_ek.hdf5',
        'test': 'pirad_erc/data_2d_from_0_to_20_size_256_256_res_0.625_0.625_ek.hdf5',
        'val': 'pirad_erc/data_2d_from_20_to_40_size_256_256_res_0.625_0.625_ek.hdf5'
        },
    'acdc': {
        'all': 'acdc/data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5',
        },
    'rvsc': {
        'all': 'rvsc/data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5',
        },
}


PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70]]

def convert_3d(masks):
    masks_3d = np.zeros([256, 256, 3])
    for i in range(256):
        for j in range(256):
            l = int(masks[i, j])
            masks_3d[i, j, :] = np.array(PALETTE[l])
    return Image.fromarray(np.uint8(masks_3d)).convert('RGB')

def convert_args(input_str):
    if input_str == 'None':
        return None
    elif isinstance(input_str, int):
        return bool(input_str)
    else:
        if ',' in input_str:
            return input_str.split(',')
        else:
            return input_str
    
def flatten_dict(cfg):
    args = {}
    for params in cfg:
        if type(cfg[params]) is dict:
            for k in cfg[params]:
                args[k] = convert_args(cfg[params][k])
        else:
            args[params] = convert_args(cfg[params])
    
    if (type(args['split']) is str):
        args['split'] = [args['split']]

    return args

def parse_config(cfg):
    cfg = OmegaConf.to_container(cfg)

    args = flatten_dict(cfg)
    pprint.pprint(args)
    
    return args

@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def start_training(cfg: DictConfig) -> None:
    args = parse_config(cfg)
    anatomy = args['anatomy']
    path = args['path']
    tgtpath = os.path.join(args['tgtpath'], anatomy)

    dataset = args['dataset']
    for split in args['split']:        
        if dataset in ['hcp1', 'hcp2', 'abide_caltech', 'abide_stanford', 'pirad_erc', 'hcp1_full', 'hcp2_full']:
            dataset_folder = DATASET_PATHS[dataset][split]
            h5_fh = h5py.File(f'{path}/{dataset_folder}', 'r')
            images_all = h5_fh['images']
            masks_all = h5_fh['labels']
        elif dataset == 'nci':
            dataset_folder = DATASET_PATHS[dataset]['all']
            h5_fh = h5py.File(f'{path}/{dataset_folder}', 'r')
            split_name = split
            if split_name == 'val':
                split_name = 'validation'

            images_all = h5_fh['images_'+ split_name]
            masks_all = h5_fh['masks_'+ split_name]

        else:
            dataset_folder = DATASET_PATHS[dataset]['all']
            h5_fh = h5py.File(f'{path}/{dataset_folder}', 'r')
            split_name = split
            if split_name == 'val':
                split_name = 'validation'

            images_all = h5_fh['images_'+ split_name]
            masks_all = h5_fh['labels_'+ split_name]

        if args['filter']:
            tgt_folder_imgs = f'{tgtpath}/{dataset}/images/{split}-filtered/'
            tgt_folder_labels = f'{tgtpath}/{dataset}/labels/{split}-filtered/'
        else:
            tgt_folder_imgs = f'{tgtpath}/{dataset}/images/{split}/'
            tgt_folder_labels = f'{tgtpath}/{dataset}/labels/{split}/'

        os.makedirs(tgt_folder_imgs, exist_ok=True) 
        os.makedirs(tgt_folder_labels, exist_ok=True) 
        
        n_images = len(images_all)
        print(dataset, split, n_images)
        pbar = tqdm(range(n_images))
        filter_count = 0
        for i_img in pbar:
            pbar.set_description(f"{tgt_folder_labels}/{i_img:04d}.png")
            img = images_all[i_img]
            masks = masks_all[i_img]            
            if args['filter']:
                if np.sum(masks) == 0:
                    save = False
                    filter_count += 1
                else:
                    save = True
            else:
                save = True

            if save:
                img_pil = Image.fromarray(np.uint8(img*255))#.convert('RGB')

                masks_pil_3d = convert_3d(masks)            
                masks_pil = Image.fromarray(np.uint8(masks))
                
                img_pil.save(f"{tgt_folder_imgs}/{i_img:04d}.png","PNG")
                masks_pil_3d.save(f"{tgt_folder_labels}/{i_img:04d}.png","PNG")
                masks_pil.save(f"{tgt_folder_labels}/{i_img:04d}_labelTrainIds.png","PNG")

        print(f'{filter_count} images out of {n_images} are empty.')
        print('done\n')

if __name__ == '__main__':
    start_training()