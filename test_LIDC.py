#
# This file is use to test and debug the setting of diffDRR used to create LIDC dataset
# 


from importlib import reload
import os
import itk
#
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms

#
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyvista
import torch
from torch.nn.functional import normalize
import torch.nn as nn
#

# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
#
from diffdrr.drr import DRR
from diffdrr import utils
#
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr, drr_to_mesh, img_to_mesh
# from diffdrr.pose import convert

#
## library to read lung cancer database
from sqlalchemy import func # required to query the db
import thirdparty.pylidc.pylidc as pl

from datetime import timedelta




# Start pyvista session 

pyvista.start_xvfb()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os#from itk import SimpleITKFiltersPython as sitk
print(os.system("which python"))

# Parmeters

visualize_on=False
N_views = 10
sdr = 300.0  ## source-to-detector radius 
height=1024 # height of output images
drr_height = 512 ### size of the projected image by drr 
#height = 1024 ### size of the projected image
delx = (4*200)/height
delx_drr = (4*200)/drr_height

focal_length=2*sdr ## distance source to sensor array
h_off = 1.0 if height % 2 else 0.5
w_off = h_off
dely = delx
dely_drr = delx_drr
cx = delx * ( height // 2 + w_off )
cy = dely * ( height // 2 + h_off )

K=np.array([[focal_length, 0.0, -cx], [0.0, focal_length, -cy], [0.,0.,1.]])

### to generate the complete dataset, the code below has to be looped over all patients IDs
pid_offset=0 ## offset of pid allow to keep already processed patients


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu" ### for height = 1024 need to test gpu with big mem
print(f"Found torch device is {device}")
#
if torch.cuda.device_count() > 1 :
    print("Using", torch.cuda.device_count(), "GPUs!")
# transform for image resizing
transform_resize = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(size=height), transforms.ToTensor() ]) 

# Take 1 patient as example
patient_list=['LIDC-IDRI-0001', ]


for idx_patient,pid in enumerate(patient_list):
    idx_patient += pid_offset
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first() # scan class instance for patient ID
    print(f"scan for patient {pid} is {scan}")


    vol = scan.to_volume().astype(np.float32) # creating numpy array with scan volume. dtype to match type of example volume in diffdrr library
    #vol = ann.scan.to_volume().astype(np.float32)
    #print(scan)
    #print(scan.pixel_spacing)
    ## Save scan in nii.gz file 
    #print(vol.shape)
    spacing_CT = np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing])

    print(f"CT voxel resolution is {spacing_CT} mm")

    # Read annotations 
    ann=scan.annotations
    print(f"number of annotations for patient {pid} are {len(ann)}")
    ### Loop over annotations for CT volumes
    bb_2d=[[] for n in range(N_views)] ## list with 2D bounding boxes for all views
    bb_3d=[] 

    # intrinsics cam #
    # Initialize the DRR module for generating synthetic X-rays
    with torch.no_grad():
        torch.cuda.empty_cache()


    ### poses of camera to create projections ### this provides the geometric constraint
    ### Diff DRR being created for a cone beam CT, lets try these
    # Set the camera pose with rotation (yaw, pitch, roll) and translation (x, y, z)
    #rotation = torch.tensor([[torch.pi, 0.0, torch.pi / 2]], device=device)
    rot_x=0.0
    rot_y=0.0
    rot_z=3*torch.pi/2
    rotation = torch.tensor([[rot_x, rot_y, rot_z]], device=device)

    bx, by, bz = torch.tensor(vol.shape) * torch.tensor(spacing_CT) / 2 ### center of the CT volume in mm
    translation = torch.tensor([[bx, by, bz]], device=device) ### TO CHECK how this defines the camera pose according to the diffDRR package
    print(translation)



    params_int_cam = [2*sdr,          
                    delx, 
                    delx, 
                    (delx*height)/2 + height/2.0, 
                    (delx*height)/2 + height/2.0 
                    ] # focus, res_x, res_y, t_x, t_y
    params_ext_cam = [] # quaternions rotations camera
    bb_3d = [] 
    bb_2d = []
    ext_cam_matrix = []  # matrix of extrinsic parameter
 

    # #
    # # Generate 2d_bbox using DRR
    # # project from 3D to 2D. 
    # #
    # for idx_ann, a in enumerate(ann):
    #     #a=ann[1]
    #     #idx_ann=1
    #     #  
    #     #print(f"annotation {idx_ann}")
    #     #print(a)
    #     #print(a.bbox())
    #     ## create volume for the annotation
    #     vol_ann = np.zeros(vol.shape, dtype=np.float32)
    #     #print(type(vol_ann[0,0,0]))
    #     #print(type(vol[0,0,0]))
    #     #mask_ann = a.boolean_mask()
    #     #
    #     #vol_ann[a.bbox()][a.boolean_mask()] = 1  ### a.bbox() contains the gold standard (3D bounding box) of the localisation of each annotation 
    #     vol_ann[a.bbox()] = 200
    #     #print(len(np.argwhere(vol_ann > 0)))
        
    #     #print(f"number of voxels in the 3D bbox is {len(np.argwhere(vol > 0))}")
    #     #print(vol_ann.max())
    #     #
    #     bb_3d.append(a.bbox_matrix())
    #     print(f'bb 3D is {bb_3d}')
    #     ##  vol_ann is a binary volume the same size as the CT images, where all pixels are zero outside the box containing the indications and 1 inside the box
    #     #
    #     ## create drr object for the volume of the anotation
    #     ## create drr object from CT
    #     drr_ann = DRR(
    #             vol_ann,   # The binary volume with the annotations
    #             #vol,
    #             spacing_CT,   # Voxel dimensions of the volume
    #             sdr=sdr,   # Source-to-detector radius (half of the source-to-detector distance)
    #             height=drr_height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    #             delx=delx_drr,    # Pixel spacing (in mm)
    #         )
        
    #     drr_ann = drr_ann.to(device)
    #     drr_ann.eval()

    #     N_views = 1
    #     for idx_view in range(N_views):  
    #         #
    #         ## generates the xray images 
    #         #
    #         rad_x = idx_view*2*torch.pi/N_views
    #         #print(rad_x)
    #         rotation = torch.tensor([[rad_x, rot_y , rot_z]], device=device) 
    #         #
            
    #         # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
    #         # For example, quaternions, rotation matrix, axis-angle, etc...
    #         with torch.no_grad():
    #             img = drr_ann(rotation=rotation, translation=translation, parameterization="euler_angles", convention="ZYX")
            
    #         # resample to output size    
    #         #transform = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(size=height), transforms.ToTensor() ]) 
    #         #img = [transform(x_) for x_ in img] 
    #         img = transform_resize(img.squeeze().cpu().detach())
            
    #         # get pixels inside
    #         img[img>0] = 1
    #         #
    #         ### here use your favorite tool to save the created xray images s.
    #         ##visualisation
    #         #       
    #         nonzero_indices = np.argwhere(img.squeeze().cpu().detach().numpy().squeeze() > 0)
    #         #print(nonzero_indices)
    #         if len(nonzero_indices) > 0:        
    #             #print(nonzero_indices)
    #             #
    #             # Calculate bounding box coordinates
    #             min_row, min_cln = np.min(nonzero_indices, axis=0)
    #             max_row, max_cln = np.max(nonzero_indices, axis=0)
    #             #
    #             # Bounding box dimensions
    #             width_bb = max_row - min_row + 1
    #             height_bb = max_cln - min_cln + 1
    #             # save bounding_box
    #             bb_2d[idx_view].append([min_row, min_cln, width_bb, height_bb, 'nodule',a.subtlety, a.internalStructure, a.calcification, a.sphericity, a.margin, a.lobulation, a.spiculation, a.texture, a.malignancy])
    #             print(f'Non-empty projection for the 3D bounding box of annotation {idx_ann} on view {idx_view}')

    #             if visualize_on:
    #                 fig, ax = plt.subplots()
    #                 ax.imshow(img.squeeze())
    #                 rect = patches.Rectangle((min_cln, min_row), height_bb, width_bb, linewidth=1, edgecolor='r', facecolor='none')
    #                 ax.add_patch(rect)
    #                 plt.show()
    #                 print(f'view {idx_view+1} : {bb_2d[idx_view]}')


    #         else:
    #             print(f'Empty projection for the 3D bounding box of annotation {idx_ann} on view {idx_view}')
            
    #         del img
    #     del vol_ann
    #     del drr_ann
    #     del rotation
        
    #     print('drr_ann deleted')


            
    #
    # Generate CT image
    # this cannot be run locally because of CUDA out of memory
    # 

    ## create drr object from CT
    drr = DRR(
        vol,      # The CT volume as a numpy array
        spacing_CT,     # Voxel dimensions of the CT
        sdr=sdr,   # Source-to-detector radius (half of the source-to-detector distance)
        height=drr_height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=delx_drr,    # Pixel spacing (in mm)
    )

    print(drr) 

    drr = drr.to(device)
    drr.eval()


    N_views = 10
    for idx_view in range(N_views):

        rad_x = idx_view*2*torch.pi/N_views
        #
        rotation = torch.tensor([[rad_x, rot_y , rot_z]], device=device) 
        # q = utils.convert(rotation=rotation, input_parameterization='euler_angles', input_convention='ZYX', output_parameterization="quaternion")
        # params_ext_cam.append(q)
        # print(f'Extrinsic for cam {idx_view}: {q}')

        # pose will return a diffdrr.RigidTransform object, which has a attribute .matrix, which is extrinsic matrix with an extra row at the bottom.
        pose = utils.convert(rotation, input_parameterization='euler_angles', input_convention='ZYX')
        ext_mat = pose.matrix
        print(f'Extrinsic matrix for cam {idx_view}: {ext_mat}')


        #
        # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
        # For example, quaternions, rotation matrix, axis-angle, etc...
        with torch.no_grad():
            print(f'rotation {rotation}')
            print(f'translation {translation}')

            # Cannot run locally 
            # img = drr(rotation=rotation, translation=translation, parameterization="euler_angles", convention="ZYX")
        
        
    del drr



