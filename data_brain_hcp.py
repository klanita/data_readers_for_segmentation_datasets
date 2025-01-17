import os
import numpy as np
import logging
import gc
import h5py
import glob
import zipfile, re
import utils
from skimage.transform import rescale

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# helper function to get paths to the image and label of a certain subject
# ===============================================================
def get_image_and_label_paths(filename,
                              protocol,
                              extraction_folder):
    
    # ========================
    # read the contents inside the top-level subject directory
    # ========================
    with zipfile.ZipFile(filename, 'r') as zfile:    
        
        # ========================
        # search for the relevant files
        # ========================
        for name in zfile.namelist():        
        
            # ========================
            # search for files inside the T1w directory
            # ========================
            if re.search(r'\/T1w/', name) != None:            
            
                # ========================
                # search for .gz files inside the T1w directory
                # ========================
                if re.search(r'\.gz$', name) != None:                        
                
                    # ========================
                    # Get the image of the desired protocol.
                    # This image is the one after the 'preFreeSurfer' preprocessing steps done by FreeSurfer
                    # For details, see https://www.ncbi.nlm.nih.gov/pubmed/23668970 Figure 9.
                    # This includes 
                    #       * correction of MR gradientnonlinearity-induced distortions
                    #       * repeated runs of T1w and T2w images are aligned with a 6 DOF rigid body transformation using FLIRT, and averaged. 
                    #       * next, the average T1w and T2w images are aligned to the MNI space template (with 0.7mm resolution for the HCP data) (“acpc alignment” step)
                    #       * next, a robust initial brain extraction (skull stripping) is performed using an initial linear (FLIRT) and non-linear (FNIRT) registration of the image to the MNI template.
                    #       * removal of readout distortion
                    #       * bias field correction
                    # ========================
                    if re.search(protocol + 'w_acpc_dc_restore_brain', name) != None:                                         
                        _imgpath = zfile.extract(name, extraction_folder) # extract the image filepath                        
                        _patname = name[:name.find('/')] # extract the patient name                        
                        
                    # ========================
                    # get the segmentation mask
                    # ========================
                    if re.search('aparc.aseg', name) != None: # segmentation mask with ~100 classes 
                        if re.search('T1wDividedByT2w_',name) == None:
                            _segpath = zfile.extract(name, extraction_folder) # extract the segmentation mask                            
                            
    return _patname, _imgpath, _segpath
                            
# ===============================================================
# helper function to count number of slices perpendicular to the coronal slices (this is fixed to the 'depth' parameter for each image volume)
# ===============================================================
def count_slices(filenames,
                 idx_start,
                 idx_end,
                 protocol,
                 preprocessing_folder,
                 depth):

    num_slices = 0
    
    for idx in range(idx_start, idx_end):
        # _, image_path, _ = get_image_and_label_paths(filenames[idx], protocol, preprocessing_folder)
        # image, _, _ = utils.load_nii(image_path)        
        # num_slices = num_slices + image.shape[1] # will append slices along axes 1
        num_slices = num_slices + depth # the number of slices along the append axis will be fixed to this number to crop out zeros
        
    return num_slices

# ===============================================================
# This function carries out all the pre-processing steps
# ===============================================================
def prepare_data(input_folder,
                 output_file,
                 idx_start,
                 idx_end,
                 protocol,
                 size,
                 depth,
                 target_resolution,
                 preprocessing_folder):

    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + '*.zip'))
    logging.info('Number of images in the dataset: %s' % str(len(filenames)))

    # =======================
    # create a new hdf5 file
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # ===============================
    # Create datasets for images and labels
    # ===============================
    data = {}
    num_slices = count_slices(filenames,
                              idx_start,
                              idx_end,
                              protocol,
                              preprocessing_folder,
                              depth)
    
    # ===============================
    # the sizes of the image and label arrays are set as: [(number of coronal slices per subject*number of subjects), size of coronal slices]
    # ===============================
    data['images'] = hdf5_file.create_dataset("images", [num_slices] + list(size), dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", [num_slices] + list(size), dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================        
    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []
    
    # ===============================      
    # initialize counters
    # ===============================        
    write_buffer = 0
    counter_from = 0
    
    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):
        
        # ==================
        # get file paths
        # ==================
        patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx],
                                                                         protocol,
                                                                         preprocessing_folder)
        
        # ============
        # read the image and normalize it to be between 0 and 1
        # ============
        image, _, image_hdr = utils.load_nii(image_path)
        image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
        
        # ==================
        # read the label file
        # ==================        
        label, _, _ = utils.load_nii(label_path)        
        label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
        label = utils.group_segmentation_classes(label) # group the segmentation classes as required
                
        # ==================
        # crop volume along z axis (as there are several zeros towards the ends)
        # ==================
        image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
        label = utils.crop_or_pad_volume_to_size_along_z(label, depth)     

        # ==================
        # collect some header info.
        # ==================
        px_list.append(float(image_hdr.get_zooms()[0]))
        py_list.append(float(image_hdr.get_zooms()[2])) # since axes 1 and 2 have been swapped
        pz_list.append(float(image_hdr.get_zooms()[1]))
        nx_list.append(image.shape[0]) 
        ny_list.append(image.shape[1]) # since axes 1 and 2 have been swapped
        nz_list.append(image.shape[2])
        pat_names_list.append(patient_name)
        
        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')
                        
        # ======================================================  
        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution[0],
                        image_hdr.get_zooms()[2] / target_resolution[1]] # since axes 1 and 2 have been swapped

        for zz in range(image.shape[2]):

            # ============
            # rescale the images and labels so that their orientation matches that of the nci dataset
            # ============            
            image2d_rescaled = rescale(np.squeeze(image_normalized[:, :, zz]),
                                                  scale_vector,
                                                  order=1,
                                                  preserve_range=True,
                                                  channel_axis=None,
                                                  mode = 'constant')
 
            label2d_rescaled = rescale(np.squeeze(label[:, :, zz]),
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  channel_axis=None,
                                                  mode='constant')
            
            # ============            
            # crop or pad to make of the same size
            # ============            
            image2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(image2d_rescaled, size[0], size[1])
            label2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(label2d_rescaled, size[0], size[1])

            # ============   
            # append to list
            # ============   
            image_list.append(image2d_rescaled_rotated_cropped)
            label_list.append(label2d_rescaled_rotated_cropped)

            # ============   
            # increment counter
            # ============   
            write_buffer += 1

            # ============   
            # Writing needs to happen inside the loop over the slices
            # ============   
            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                     image_list,
                                     label_list,
                                     counter_from,
                                     counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # ============   
                # update counters 
                # ============   
                counter_from = counter_to
                write_buffer = 0
        
    # ============   
    # write leftover data
    # ============   
    logging.info('Writing remaining data')
    counter_to = counter_from + write_buffer
    _write_range_to_hdf5(data,
                         image_list,
                         label_list,
                         counter_from,
                         counter_to)
    _release_tmp_memory(image_list,
                        label_list)

    # ============   
    # Write the small datasets - image resolutions, sizes, patient ids
    # ============   
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    
    # ============   
    # close the hdf5 file
    # ============   
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    hdf5_data['images'][counter_from : counter_to, ...] = img_arr
    hdf5_data['labels'][counter_from : counter_to, ...] = lab_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):

    img_list.clear()
    mask_list.clear()
    gc.collect()
    
# ===============================================================
# function to read a single subjects image and labels without any pre-processing
#  returns an image volume as [nx, ny, nz], where
#      [nx, ny] is the image size of the coronal slices (kept the same as in the original images)
#      nz id the number of coronal slices (fixed by cropping / padding to 'depth)
# the image are normalized to [0-1].
# the segmentation labels are grouped into 15 classes.
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    idx,
                                    protocol,
                                    preprocessing_folder,
                                    depth):
    
    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + '*.zip'))

    # ==================
    # get file paths
    # ==================
    patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx],
                                                                     protocol,
                                                                     preprocessing_folder)
    
    # ============
    # read the image and normalize it to be between 0 and 1
    # ============
    image, _, image_hdr = utils.load_nii(image_path)
    image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    image = utils.normalise_image(image, norm_type='div_by_max')
    
    # ==================
    # read the label file
    # ==================        
    label, _, _ = utils.load_nii(label_path)        
    label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    label = utils.group_segmentation_classes(label) # group the segmentation classes as required
    
    # ==================
    # crop volume along z axis (as there are several zeros towards the ends)
    # ==================
    image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
    label = utils.crop_or_pad_volume_to_size_along_z(label, depth)   
    
    return image, label

# ===============================================================
# Main function that preprocesses images and labels and returns a handle to a hdf5 file containing them.
# The images and labels are returned as [num_subjects*nz, nx, ny],
#   where nz is the number of coronal slices per subject ('depth')
#         nx, ny is the size of the coronal slices ('size')  
#         the resolution in the coronal slices is 'target_resolution'
#         the resolution perpendicular to coronal slices is unchanged.
# The read images are the ones from after the 'PreFreeSurfer' preprocessing pipeline.
    # So, they have undergone several pre-processing steps (including skull stripping and bias field correction)
    # For details, see the comments in the get_image_and_label_paths function.
# Each image volume is normalized to [0-1].
# The segmentation labels are grouped into 15 classes.    
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                idx_start,
                                idx_end,
                                protocol,
                                size,
                                depth,
                                target_resolution,
                                force_overwrite=False):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_2d_size_%s_depth_%d_res_%s_from_%d_to_%d.hdf5' % (protocol, size_str, depth, res_str, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     idx_start,
                     idx_end,
                     protocol,
                     size,
                     depth,
                     target_resolution,
                     preprocessing_folder)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')
