import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel.orientations as nio
import nibabel.affines as nia
import nibabel.processing as nip
from PIL import Image
from skimage.transform import rescale
import pickle
from tqdm import tqdm

PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70]]

def get_subfolders(directory):
    """
    Returns a list of names of immediate subfolders in the given directory.
    """
    return sorted([name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))])

def nib_to_numpy(img: nib.Nifti1Image) -> np.ndarray:
    """Convert nibabel image to numpy array."""
    return np.asarray(img.dataobj)

def reorient_to(
    img: nib.Nifti1Image, axcodes_to=("P", "I", "R"), verb=False
) -> nib.Nifti1Image:
    """Reorients the nifti from its original orientation to another specified orientation.

    Args:
        img (nib.Nifti1Image): nibabel image
        axcodes_to (tuple, optional): a tuple of 3 characters specifying the desired orientation. Defaults to ("P", "I", "R").
        verb (bool, optional): if True prints some debug output. Defaults to False.

    Returns:
        nib.Nifti1Image: The reoriented nibabel image
    """
    aff = img.affine
    img_np = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)  # type: ignore[attr-defined]
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    img_np = nio.apply_orientation(img_np, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, img_np.shape)
    newaff = np.matmul(aff, aff_trans)
    img = nib.Nifti1Image(img_np, newaff)
    if verb:
        print(
            "[*] Image reoriented from",
            nio.ornt2axcodes(ornt_fr),
            "to",
            axcodes_to,
        )
    return img


def resample_nib(
    img: nib.Nifti1Image,
    voxel_spacing=(1, 1, 1),
    order=3,
    cval=-1024,
    verb=False,
) -> nib.Nifti1Image:
    """Resamples the nifti from its original spacing to another specified spacing.

    Args:
        img (nib.Nifti1Image): nibabel image
        voxel_spacing (tuple, optional): a tuple of 3 integers specifying the desired new spacing. Defaults to (1, 1, 1).
        order (int, optional): the order of interpolation. Defaults to 3.
        cval (int, optional): cval used in nip.resample_from_to. Defaults to -1024.
        verb (bool, optional): if true prints some debug output. Defaults to False.

    Returns:
        nib.Nifti1Image: The resampled nibabel image
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()  # type: ignore[attr-defined]
    # Calculate new shape
    new_shp = tuple(
        np.rint(
            [
                shp[0] * zms[0] / voxel_spacing[0],
                shp[1] * zms[1] / voxel_spacing[1],
                shp[2] * zms[2] / voxel_spacing[2],
            ]
        ).astype(int)
    )
    new_aff = nia.rescale_affine(aff, shp, voxel_spacing, new_shp)
    img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=cval)
    if verb:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return img

def resample_reorient_to(
    mask: nib.Nifti1Image, orientation: tuple, spacing: tuple
) -> nib.Nifti1Image:
    """Resample and reorient mask to orientation and spacing.

    Args:
        mask (nib.Nifti1Image): mask to be resampled/reoriented
        orientation (tuple): orientation in axis codes
        spacing (tuple): resolution in mm

    Returns:
        nib.Nifti1Image: resampled/reoriented mask
    """
    if nio.aff2axcodes(mask.affine) != orientation:
        mask = reorient_to(mask, orientation)

    assert all(
        x == y
        for x, y in zip(
            nio.ornt2axcodes(nio.io_orientation(mask.affine)),
            orientation,
        )
    ), "Orientation not correct. Expected orientation {}, got {}".format(
        orientation, nio.ornt2axcodes(nio.io_orientation(mask.affine))
    )

    if not np.isclose(mask.header.get_zooms(), spacing).all():  # type: ignore[attr-defined]
        mask = resample_nib(mask, spacing, order=0, cval=0)

    assert all(
        np.float32(x) == np.float32(y)
        for x, y in zip(mask.header.get_zooms(), spacing)  # type: ignore[attr-defined]
    ), "Spacing not correct. Expected spacing {}, got {}".format(
        spacing, mask.header.get_zooms()  # type: ignore[attr-defined]
    )

    return mask

def clean_labels(patient_mask_np, relabel_dict):
    """Relabels the mask according to the relabel_dict.

    Args:
        patient_mask_np (np.ndarray): mask of the patient
        relabel_dict (dict): dictionary to relabel the mask

    Returns:
        np.ndarray: relabeled mask
    """
    patient_mask_np_clean = np.copy(patient_mask_np)
    patient_labes = np.unique(patient_mask_np)
    for label in patient_labes:
        if not (label in relabel_dict.keys()):
            patient_mask_np_clean[patient_mask_np_clean == label] = 0
        
    for old_label, new_label in relabel_dict.items():
        patient_mask_np_clean[patient_mask_np_clean == old_label] = new_label
    return patient_mask_np_clean

def find_bounding_boxes(segmentation_array):
    bounding_boxes = {}  # Dictionary to hold bounding boxes for each label
    unique_labels = np.unique(segmentation_array)[1:]  # Exclude background label (0)

    # for label in unique_labels:
    # Find the indexes where the current label is found
    locations = np.where(segmentation_array != 0)
    
    # Determine the bounding box coordinates for the current label
    min_z, max_z = locations[0].min(), locations[0].max()
    min_y, max_y = locations[1].min(), locations[1].max()
    min_x, max_x = locations[2].min(), locations[2].max()
    
    # Save the bounding box (min and max coordinates for each dimension)
    bounding_boxes = ((min_z, min_y, min_x), (max_z, max_y, max_x))
    
    return bounding_boxes

def crop_image_to_center(image, masks, target_dims, bounding_box):
    # Unpack the target dimensions
    target_d, target_h, target_w = target_dims
    
    # Calculate the center of the bounding box
    bbox_center_z = (bounding_box[0][0] + bounding_box[1][0]) // 2
    bbox_center_y = (bounding_box[0][1] + bounding_box[1][1]) // 2
    bbox_center_x = (bounding_box[0][2] + bounding_box[1][2]) // 2
    
    # Calculate start and end indices for cropping
    start_z = max(0, bbox_center_z - target_d // 2)
    start_y = max(0, bbox_center_y - target_h // 2)
    start_x = max(0, bbox_center_x - target_w // 2)
    
    # Adjust the start indices if necessary to fit the target size
    end_z = start_z + target_d
    end_y = start_y + target_h
    end_x = start_x + target_w
    
    # Ensure the crop dimensions do not exceed the original image size
    original_d, original_h, original_w = image.shape
    if end_z > original_d:
        end_z = original_d
        start_z = original_d - target_d
    if end_y > original_h:
        end_y = original_h
        start_y = original_h - target_h
    if end_x > original_w:
        end_x = original_w
        start_x = original_w - target_w
    
    # Crop the image
    cropped_image = image[start_z:end_z, start_y:end_y, start_x:end_x]
    cropped_masks = masks[start_z:end_z, start_y:end_y, start_x:end_x]
    
    return cropped_image, cropped_masks

def normalise_image(image, norm_type = 'div_by_max'):
    '''
    make image zero mean and unit standard deviation
    '''
    if norm_type == 'zero_mean':
        img_o = np.float32(image.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        normalized_img = np.divide((img_o - m), s)
        
    elif norm_type == 'div_by_max':
        perc1 = np.percentile(image,1)
        perc99 = np.percentile(image,99)
        normalized_img = np.divide((image - perc1), (perc99 - perc1))
        normalized_img[normalized_img < 0] = 0.0
        normalized_img[normalized_img > 1] = 1.0
    
    return normalized_img

def read_nifti_file(patient_file, resample_resolution=(1, 1, 1)):    
    orientation=("P", "L", "I")
    patient_data = nib.load(patient_file)
    patient_data = resample_reorient_to(
                patient_data, orientation, resample_resolution
            )
    patient_data_np = nib_to_numpy(patient_data)
    return patient_data_np

def center_crop(img, new_width=256, new_height=256):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def convert_3d(masks):
    masks_3d = np.zeros([256, 256, 3])
    for i in range(256):
        for j in range(256):
            l = int(masks[i, j])
            masks_3d[i, j, :] = np.array(PALETTE[l])
    return Image.fromarray(np.uint8(masks_3d)).convert('RGB')


def convert_patient_per_slice(patient_data_np, patient_mask_np, patinet_name, tgt_folder_imgs, tgt_folder_labels):
    for slice_id in range(patient_data_np.shape[-1]):
        img_pil = Image.fromarray(np.uint8(center_crop(patient_data_np[:, :, slice_id]*255)))
        masks_cropped = center_crop(patient_mask_np[:, :, slice_id])
        masks_pil = Image.fromarray(np.uint8(masks_cropped))
        masks_pil_3d = convert_3d(masks_cropped)

        img_pil.save(f"{tgt_folder_imgs}/{patinet_name}_{slice_id:04d}.png","PNG")
        masks_pil_3d.save(f"{tgt_folder_labels}/{patinet_name}_{slice_id:04d}.png","PNG")
        masks_pil.save(f"{tgt_folder_labels}/{patinet_name}_{slice_id:04d}_labelTrainIds.png","PNG")

def get_cropped_volume(patient_file, patient_file_masks):
    relabel_dict = {0:0, 1:1, 3:2, 5:3, 7:4, 9:5}
    image_data = nib.load(patient_file)
    original_spacing = np.array(image_data.header.get_zooms()[:3])
    target_spacing = [0.7, original_spacing[0], 0.7]
    image = read_nifti_file(patient_file, target_spacing)
    masks = read_nifti_file(patient_file_masks, target_spacing)
    masks = clean_labels(masks, relabel_dict)
    mid = image.shape[1] // 2
    print('Depth before', image.shape[1])
    image_norm = normalise_image(image)
    image_norm = np.swapaxes(image_norm, 1, 2)
    masks = np.swapaxes(masks, 1, 2)
    bounding_boxes = find_bounding_boxes(masks)
    # depth = image_norm.shape[0]
    depth = 12
    target_dims = (256, 256, depth)  # Height, Width, Depth
    cropped_image, cropped_masks = crop_image_to_center(image_norm, masks, target_dims, bounding_boxes)
    print('Depth after', cropped_image.shape[-1])

    meta_info = dict()
    meta_info['px'] = original_spacing[1]
    meta_info['py'] = original_spacing[2]
    meta_info['pz'] = original_spacing[0]
    meta_info['resolution_proc'] = target_spacing

    return cropped_image, cropped_masks, meta_info

def get_CT_verse_data(input_image_path, input_masks_path, depth=120):
    image_data = nib.load(input_image_path)
    original_spacing = np.array(image_data.header.get_zooms()[:3])
    target_spacing = [0.7, original_spacing[-1], 0.7]
    print('original_spacing:', original_spacing)
    # original_shape = np.array(image_data.get_fdata().shape)
    # print('original_shape:', original_shape)

    image = read_nifti_file(input_image_path, target_spacing)
    masks = read_nifti_file(input_masks_path, target_spacing)
    relabel_dict = {20:1, 21:2, 22:3, 23:4, 24:5}
    patient_mask_np = clean_labels(masks, relabel_dict)

    image_norm = normalise_image(image)
    # mid = masks.shape[1] // 2
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image_norm[:, mid, : ], cmap='gray')
    # axs[1].imshow(patient_mask_np[:, mid, : ])

    # make depth along last dim
    image_norm = np.swapaxes(image_norm, 1, 2)
    patient_mask_np = np.swapaxes(patient_mask_np, 1, 2)
    bounding_boxes = find_bounding_boxes(patient_mask_np)
    # filter out empty masks
    z_dim = bounding_boxes[1][-1] - bounding_boxes[0][-1]
    print('Depth before', image_norm.shape[-1])
    if image_norm.shape[-1] < depth:
        return None, None, None

    print('Depth optimal', z_dim)
    target_dims = (256, 256, depth)  # Depth, Height, Width
    cropped_image, cropped_masks = crop_image_to_center(image_norm, patient_mask_np, target_dims, bounding_boxes)
    print('Depth after', cropped_image.shape[-1])

    mid = cropped_masks.shape[2] // 2
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(cropped_image[:, :, mid ], cmap='gray')
    # axs[1].imshow(cropped_masks[:, :, mid ])

    meta_info = dict()
    meta_info['px'] = original_spacing[0]
    meta_info['py'] = original_spacing[1]
    meta_info['pz'] = original_spacing[-1]
    meta_info['resolution_proc'] = target_spacing

    return cropped_image, cropped_masks, meta_info

def get_file_name(directory):
    """
    Returns a list of names of immediate subfolders in the given directory.
    """
    files = [f for f in os.listdir(directory) if ('._' not in f) and ('.nii.gz' in f)]
    return os.path.join(directory, files[0])

def get_slices(dataset, folder, split):       
    tgt_path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/lumbarspine/'

    if dataset == 'VerSe':
        path = '/usr/bmicnas02/data-biwi-01/lumbarspine/VerSe-subject-with-ct/VerSe-subject/'
        dataset_folder = f'{path}/{folder}/rawdata/'
            
        masks_folder = f'{path}/{folder}/derivatives_full/'

    else:
        path = '/usr/bmicnas02/data-biwi-01/lumbarspine/datasets_lara/'

        dataset_folder = f'{path}{dataset}/{folder}/rawdata/'
        print(dataset_folder)
            
        if dataset == 'Dataset7':
            masks_folder = f'{path}{dataset}/{folder}/segmentations_full/'
        else:
            masks_folder = f'{path}{dataset}/{folder}/derivatives_full/'
        
    patients = get_subfolders(dataset_folder)
    # print(len(patients), 'patients found')

    num_slices_per_patient = []
    num_patients = len(patients)
    # num_patients = 1
    pbar = tqdm(range(94, num_patients))
    meta_info = dict()
    for i in pbar:
        # if i <= num_patients // 2:
        #     split = 'train'
        # else:
        #     split = 'test'

        patinet_name = patients[i]
        pbar.set_description(f'[{i}][{patinet_name}]')

        if dataset == 'Dataset7':
            tgt_folder_name = 'Dataset7V'
            patient_file = f'{dataset_folder}{patients[i]}/Img_{patients[i]}.nii.gz'
            patient_mask = f'{masks_folder}{patients[i]}/Img_{patients[i]}_seg.nii.gz'
            patient_data_np, patient_mask_np, meta_info_patient = get_cropped_volume(patient_file, patient_mask)

        elif dataset == 'MRI_dataset-256':
            tgt_folder_name = 'MRSpineSegV'      
            try:          
                patient_file = f'{dataset_folder}{patients[i]}/img_{patients[i]}_t2.nii.gz'
                patient_mask = f'{masks_folder}{patients[i]}/img_{patients[i]}_seg.nii.gz'
                patient_data_np, patient_mask_np, meta_info_patient = get_cropped_volume(patient_file, patient_mask)
            except:
                patient_file = f'{dataset_folder}{patients[i]}/MRSpineSeg_{patients[i]}_0000.nii.gz'
                patient_mask = f'{masks_folder}{patients[i]}/MRSpineSeg_mask_{patients[i]}.nii.gz'
                patient_data_np, patient_mask_np, meta_info_patient = get_cropped_volume(patient_file, patient_mask)
        elif dataset == 'VerSe':
            tgt_folder_name = 'VerSe'
            subject = patients[i]
            patient_file = get_file_name(f'{dataset_folder}/{subject}/')
            patient_mask = get_file_name(f'{masks_folder}/{subject}/')
            
            patient_data_np, patient_mask_np, meta_info_patient = get_CT_verse_data(patient_file, patient_mask)
        
        if not (patient_data_np is None):
            for key, value in meta_info_patient.items():
                if key in meta_info:
                    meta_info[key].append(value)
                else:
                    meta_info[key] = [value]
            # print('patient_data_np:', patient_data_np.min(), patient_data_np.max())
            # print(patient_mask_np.shape, patient_data_np.shape)
            # print(np.unique(patient_mask_np))
            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # mid = patient_data_np.shape[-1] // 2
            # axs[0].imshow(patient_data_np[:, :, mid], cmap='gray')
            # axs[1].imshow(patient_mask_np[:, :, mid])

            # print(len(patients)*patient_data_np.shape[1], 'slices')        
            num_slices_per_patient.append(patient_data_np.shape[-1])

            tgt_folder_imgs = f'{tgt_path}/{tgt_folder_name}/images/{split}/'
            tgt_folder_labels = f'{tgt_path}/{tgt_folder_name}/labels/{split}/'
            print(tgt_folder_imgs)
            
            os.makedirs(tgt_folder_imgs, exist_ok=True)
            os.makedirs(tgt_folder_labels, exist_ok=True)

            convert_patient_per_slice(patient_data_np, patient_mask_np, patinet_name, 
                                    tgt_folder_imgs, tgt_folder_labels)

            pbar.set_description(f'[{patinet_name}], Slices {patient_data_np.shape[-1]}')
        else:
            pbar.set_description(f'[{patinet_name}], No slices')
    
    meta_file_path = f'{tgt_path}/{tgt_folder_name}/scale_{split}.pickle'
    with open(meta_file_path, 'wb') as handle:
        meta_info['resolution_proc'] = np.array(meta_info['resolution_proc'][0])
        pickle.dump(meta_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(meta_file_path, 'wb') as handle:
    #     pickle.dump(meta_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == "__main__":
    get_slices('VerSe', '01_training', 'train')
    # get_slices('VerSe', '03_test', 'test')
    # get_slices('VerSe', '02_validation', 'val')