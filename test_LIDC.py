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


    params_int_cam = [2*sdr,          
                    delx, 
                    delx, 
                    (delx*height)/2 + height/2.0, 
                    (delx*height)/2 + height/2.0 
                    ] # focus, res_x, res_y, t_x, t_y
    params_ext_cam = [] # quaternions rotations camera
    bb_3d = [] 
    bb_2d = []


            
    #
    N_views = 1
    for idx_view in range(N_views):

        rad_x = idx_view*2*torch.pi/N_views
        #
        rotation = torch.tensor([[rad_x, rot_y , rot_z]], device=device) 
        q = utils.convert(rotation=rotation, input_parameterization='euler_angles', input_convention='ZYX', output_parameterization="quaternion")
        #print(q)
        params_ext_cam.append(q)
        print(f'Extrinsic for cam {idx_view}: {q}')
        #
        # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
        # For example, quaternions, rotation matrix, axis-angle, etc...
        with torch.no_grad():
            print(f'rotation {rotation}')
            print(f'translation {translation}')

            # Cannot run locally 
            # img = drr(rotation=rotation, translation=translation, parameterization="euler_angles", convention="ZYX")
        
        
    del drr



