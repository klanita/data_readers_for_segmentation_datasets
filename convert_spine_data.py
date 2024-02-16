import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel.orientations as nio
import nibabel.affines as nia
import nibabel.processing as nip
from PIL import Image

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

def read_nifti_file(patient_file):
    resample_resolution=(1, 1, 1)
    orientation=("P", "L", "I")
    patient_data = nib.load(patient_file)
    patient_data = resample_reorient_to(
                patient_data, orientation, resample_resolution
            )
    patient_data_np = nib_to_numpy(patient_data)
    print(patient_data_np.shape)
    return patient_data_np

def get_slice(i, dataset, folder):
    dataset_folder = f'{path}{dataset}/{folder}/rawdata/'
    masks_folder = f'{path}{dataset}/{folder}/derivatives_full/'
    patients = get_subfolders(dataset_folder)
    print(len(patients), 'patients found')
    
    if dataset== 'MRI_dataset-256':
        patient_file = f'{dataset_folder}{patients[i]}/img_{patients[i]}_t2.nii.gz'
        patient_mask = f'{masks_folder}{patients[i]}/img_{patients[i]}_seg.nii.gz'
    else:
        patient_file = f'{dataset_folder}{patients[i]}/MRSpineSeg_{patients[i]}_0000.nii.gz'
        patient_mask = f'{masks_folder}{patients[i]}/MRSpineSeg_mask_{patients[i]}.nii.gz'

    patient_data_np = read_nifti_file(patient_file)
    patient_mask_np = read_nifti_file(patient_mask)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(patient_data_np[:, 40, :], cmap='gray')
    axs[1].imshow(patient_mask_np[:, 40, :], cmap='gray')

    print(len(patients)*patient_data_np.shape[1], 'slices')
    
    patient_data_np = patient_data_np / patient_data_np.max()
    return patient_data_np, patient_mask_np