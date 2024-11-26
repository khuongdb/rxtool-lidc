# %% imports
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
import torch
from torch.nn.functional import normalize
import torch.nn as nn
#

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
#
from diffdrr.drr import DRR
from diffdrr import utils
#
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr
#
## library to read lung cancer database
from sqlalchemy import func # required to query the db
import thirdparty.pylidc.pylidc as pl

from datetime import timedelta



# %%
import os#from itk import SimpleITKFiltersPython as sitk
print(os.system("which python"))

# %% [markdown]
# ### Read Lung Cancer Database files
# Follow instructions at:
# 
# https://pylidc.github.io/
# 
# Please note the folder for the LDCT Database must be set in file  "/home/[user]/.pylidcrc".
# 
# When running the code below, if getting errors such as np.bool not found or np.int not found, change the sources as follows
# 
# - replace *np.int* by *int*
# 
# - replace *np.bool* by *bool*

# %%
# Folder with Lung Cancer Dataset of CT volumes 
LIDC_DBB_folder="/gpfs_new/cold-data/InputData/rxnum/public_datasets/LungCancerCT/LIDC/"
# create .pylidcrc file in ~/
os.system("echo [dicom] > ~/.pylidcrc")
os.system(f'echo path={LIDC_DBB_folder} >> ~/.pylidcrc')
os.system("echo warn=True >> ~/.pylidcrc")
#pl = reload(pl)

# %%
# Folder to store output data 
#output_folder='/gpfs_new/scratch/unites_org/dst/tsi/NDIS/FilRouge/MV2D/Data_For_NuScenes'
output_folder = '/gpfs_new/scratch/users/jbetancur/rxnum/Data_For_NuScenes'
output_folder_cams = os.path.join(output_folder,'Cams/')
output_folder_images = os.path.join(output_folder,'Images/')
output_folder_labels2d = os.path.join(output_folder,'Labels2d/')
output_folder_labels3d = output_folder
output_folder_volume=os.path.join(output_folder,'CT/')
filename_patient_processed = os.path.join(output_folder,'patients_processed.txt')


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
patient_list=['LIDC-IDRI-0001', 'LIDC-IDRI-0097', 'LIDC-IDRI-0168', 'LIDC-IDRI-0240', 'LIDC-IDRI-0311', 'LIDC-IDRI-0382', 'LIDC-IDRI-0453', 'LIDC-IDRI-0595', 'LIDC-IDRI-0666', 'LIDC-IDRI-0737', 'LIDC-IDRI-0808', 'LIDC-IDRI-0879', 'LIDC-IDRI-0950',            
            'LIDC-IDRI-0002', 'LIDC-IDRI-0098', 'LIDC-IDRI-0169', 'LIDC-IDRI-0241', 'LIDC-IDRI-0312', 'LIDC-IDRI-0383', 'LIDC-IDRI-0454', 'LIDC-IDRI-0596', 'LIDC-IDRI-0667', 'LIDC-IDRI-0738', 'LIDC-IDRI-0809', 'LIDC-IDRI-0880', 'LIDC-IDRI-0951',
            'LIDC-IDRI-0003', 'LIDC-IDRI-0099', 'LIDC-IDRI-0170', 'LIDC-IDRI-0242', 'LIDC-IDRI-0313', 'LIDC-IDRI-0384', 'LIDC-IDRI-0455', 'LIDC-IDRI-0597', 'LIDC-IDRI-0668', 'LIDC-IDRI-0739', 'LIDC-IDRI-0810', 'LIDC-IDRI-0881', 'LIDC-IDRI-0952',
            'LIDC-IDRI-0004', 'LIDC-IDRI-0100', 'LIDC-IDRI-0171', 'LIDC-IDRI-0243', 'LIDC-IDRI-0314', 'LIDC-IDRI-0385', 'LIDC-IDRI-0456', 'LIDC-IDRI-0598', 'LIDC-IDRI-0669', 'LIDC-IDRI-0740', 'LIDC-IDRI-0811', 'LIDC-IDRI-0882', 'LIDC-IDRI-0953',
            'LIDC-IDRI-0005', 'LIDC-IDRI-0101', 'LIDC-IDRI-0172', 'LIDC-IDRI-0244', 'LIDC-IDRI-0315', 'LIDC-IDRI-0386', 'LIDC-IDRI-0457', 'LIDC-IDRI-0599', 'LIDC-IDRI-0670', 'LIDC-IDRI-0741', 'LIDC-IDRI-0812', 'LIDC-IDRI-0883', 'LIDC-IDRI-0954',
            'LIDC-IDRI-0006', 'LIDC-IDRI-0102', 'LIDC-IDRI-0173', 'LIDC-IDRI-0245', 'LIDC-IDRI-0316', 'LIDC-IDRI-0387', 'LIDC-IDRI-0458', 'LIDC-IDRI-0600', 'LIDC-IDRI-0671', 'LIDC-IDRI-0742', 'LIDC-IDRI-0813', 'LIDC-IDRI-0884', 'LIDC-IDRI-0955',
            'LIDC-IDRI-0007', 'LIDC-IDRI-0103', 'LIDC-IDRI-0174', 'LIDC-IDRI-0246', 'LIDC-IDRI-0317', 'LIDC-IDRI-0388', 'LIDC-IDRI-0459', 'LIDC-IDRI-0601', 'LIDC-IDRI-0672', 'LIDC-IDRI-0743', 'LIDC-IDRI-0814', 'LIDC-IDRI-0885', 'LIDC-IDRI-0956',
            'LIDC-IDRI-0008', 'LIDC-IDRI-0104', 'LIDC-IDRI-0175', 'LIDC-IDRI-0247', 'LIDC-IDRI-0318', 'LIDC-IDRI-0389', 'LIDC-IDRI-0460', 'LIDC-IDRI-0602', 'LIDC-IDRI-0673', 'LIDC-IDRI-0744', 'LIDC-IDRI-0815', 'LIDC-IDRI-0886', 'LIDC-IDRI-0957',
            'LIDC-IDRI-0009', 'LIDC-IDRI-0105',             
            'LIDC-IDRI-0248', 'LIDC-IDRI-0319', 'LIDC-IDRI-0390', 'LIDC-IDRI-0461', 'LIDC-IDRI-0603', 'LIDC-IDRI-0674', 'LIDC-IDRI-0745', 'LIDC-IDRI-0816', 'LIDC-IDRI-0887', 'LIDC-IDRI-0958',
            'LIDC-IDRI-0010', 'LIDC-IDRI-0106',         
            #'LIDC-IDRI-0176', #problem with data (length data not as expected)
            #'LIDC-IDRI-0177', #problem with data (length data not as expected) 
            'LIDC-IDRI-0249', 
            'LIDC-IDRI-0320', 'LIDC-IDRI-0391', 'LIDC-IDRI-0462', 'LIDC-IDRI-0604', 'LIDC-IDRI-0675', 'LIDC-IDRI-0746', 'LIDC-IDRI-0817', 'LIDC-IDRI-0888', 'LIDC-IDRI-0959',
            'LIDC-IDRI-0011', 'LIDC-IDRI-0107', 'LIDC-IDRI-0178', 'LIDC-IDRI-0250', 'LIDC-IDRI-0321', 'LIDC-IDRI-0392', 'LIDC-IDRI-0463', 'LIDC-IDRI-0605', 'LIDC-IDRI-0676', 'LIDC-IDRI-0747', 'LIDC-IDRI-0818', 'LIDC-IDRI-0889', 'LIDC-IDRI-0960',
            'LIDC-IDRI-0012', 'LIDC-IDRI-0108', 'LIDC-IDRI-0179', 'LIDC-IDRI-0251', 'LIDC-IDRI-0322', 'LIDC-IDRI-0393', 'LIDC-IDRI-0464', 'LIDC-IDRI-0606', 'LIDC-IDRI-0677', 'LIDC-IDRI-0748', 'LIDC-IDRI-0819', 'LIDC-IDRI-0890', 'LIDC-IDRI-0961',
            'LIDC-IDRI-0013', 'LIDC-IDRI-0109', 'LIDC-IDRI-0180', 'LIDC-IDRI-0252', 'LIDC-IDRI-0323', 'LIDC-IDRI-0394', 'LIDC-IDRI-0465', 'LIDC-IDRI-0607', 'LIDC-IDRI-0678', 'LIDC-IDRI-0749', 'LIDC-IDRI-0820', 'LIDC-IDRI-0891', 'LIDC-IDRI-0962',
            'LIDC-IDRI-0014', 'LIDC-IDRI-0110', 'LIDC-IDRI-0181', 'LIDC-IDRI-0253', 'LIDC-IDRI-0324', 'LIDC-IDRI-0395', 'LIDC-IDRI-0466', 'LIDC-IDRI-0608', 'LIDC-IDRI-0679', 'LIDC-IDRI-0750', 'LIDC-IDRI-0821', 'LIDC-IDRI-0892', 'LIDC-IDRI-0963',
            'LIDC-IDRI-0015', 'LIDC-IDRI-0111', 'LIDC-IDRI-0182', 'LIDC-IDRI-0254', 'LIDC-IDRI-0325', 'LIDC-IDRI-0396', 'LIDC-IDRI-0467', 'LIDC-IDRI-0609', 'LIDC-IDRI-0680', 'LIDC-IDRI-0751', 'LIDC-IDRI-0822', 'LIDC-IDRI-0893', 'LIDC-IDRI-0964',
            'LIDC-IDRI-0016', 'LIDC-IDRI-0112', 'LIDC-IDRI-0183', 'LIDC-IDRI-0255', 'LIDC-IDRI-0326', 'LIDC-IDRI-0397', 'LIDC-IDRI-0468', 'LIDC-IDRI-0610', 'LIDC-IDRI-0681', 'LIDC-IDRI-0752', 'LIDC-IDRI-0823', 'LIDC-IDRI-0894', 'LIDC-IDRI-0965',
            'LIDC-IDRI-0017', 'LIDC-IDRI-0113', 'LIDC-IDRI-0184', 'LIDC-IDRI-0256', 'LIDC-IDRI-0327', 'LIDC-IDRI-0398', 'LIDC-IDRI-0469', 'LIDC-IDRI-0611', 'LIDC-IDRI-0682', 'LIDC-IDRI-0753', 'LIDC-IDRI-0824', 'LIDC-IDRI-0895', 'LIDC-IDRI-0966',
            'LIDC-IDRI-0018', 'LIDC-IDRI-0114', 'LIDC-IDRI-0185', 'LIDC-IDRI-0257', 'LIDC-IDRI-0328', 'LIDC-IDRI-0399', 'LIDC-IDRI-0470', 'LIDC-IDRI-0612', 'LIDC-IDRI-0683', 'LIDC-IDRI-0754', 'LIDC-IDRI-0825', 'LIDC-IDRI-0896', 'LIDC-IDRI-0967',
            'LIDC-IDRI-0019', 'LIDC-IDRI-0115', 'LIDC-IDRI-0186', 'LIDC-IDRI-0258', 'LIDC-IDRI-0329', 'LIDC-IDRI-0400', 'LIDC-IDRI-0471', 'LIDC-IDRI-0613', 'LIDC-IDRI-0684', 'LIDC-IDRI-0755', 'LIDC-IDRI-0826', 'LIDC-IDRI-0897', 'LIDC-IDRI-0968',
            'LIDC-IDRI-0020', 'LIDC-IDRI-0116', 'LIDC-IDRI-0187', 'LIDC-IDRI-0259', 'LIDC-IDRI-0330', 'LIDC-IDRI-0401', 'LIDC-IDRI-0472', 'LIDC-IDRI-0614', 'LIDC-IDRI-0685', 'LIDC-IDRI-0756', 'LIDC-IDRI-0827', 'LIDC-IDRI-0898', 'LIDC-IDRI-0969',
            'LIDC-IDRI-0021', 'LIDC-IDRI-0117', 'LIDC-IDRI-0188', 'LIDC-IDRI-0260', 'LIDC-IDRI-0331', 'LIDC-IDRI-0402', 'LIDC-IDRI-0473', 'LIDC-IDRI-0615', 'LIDC-IDRI-0686', 'LIDC-IDRI-0757', 'LIDC-IDRI-0828', 'LIDC-IDRI-0899', 'LIDC-IDRI-0970',
            'LIDC-IDRI-0022', 'LIDC-IDRI-0118', 'LIDC-IDRI-0189', 'LIDC-IDRI-0261', 'LIDC-IDRI-0332', 'LIDC-IDRI-0403', 'LIDC-IDRI-0474', 'LIDC-IDRI-0616', 'LIDC-IDRI-0687', 'LIDC-IDRI-0758', 'LIDC-IDRI-0829', 'LIDC-IDRI-0900', 'LIDC-IDRI-0971',
            'LIDC-IDRI-0023', 'LIDC-IDRI-0119', 'LIDC-IDRI-0190', 'LIDC-IDRI-0262', 'LIDC-IDRI-0333', 'LIDC-IDRI-0404', 'LIDC-IDRI-0475', 'LIDC-IDRI-0617', 'LIDC-IDRI-0688', 'LIDC-IDRI-0759', 'LIDC-IDRI-0830', 'LIDC-IDRI-0901', 'LIDC-IDRI-0972',
            'LIDC-IDRI-0024', 'LIDC-IDRI-0120', 'LIDC-IDRI-0191', 'LIDC-IDRI-0263', 'LIDC-IDRI-0334', 'LIDC-IDRI-0405', 'LIDC-IDRI-0476', 'LIDC-IDRI-0618', 'LIDC-IDRI-0689', 'LIDC-IDRI-0760', 'LIDC-IDRI-0831', 'LIDC-IDRI-0902', 'LIDC-IDRI-0973',
            'LIDC-IDRI-0025', 'LIDC-IDRI-0121', 'LIDC-IDRI-0192', 'LIDC-IDRI-0264', 'LIDC-IDRI-0335', 'LIDC-IDRI-0406', 'LIDC-IDRI-0477', 'LIDC-IDRI-0619', 'LIDC-IDRI-0690', 'LIDC-IDRI-0761', 'LIDC-IDRI-0832', 'LIDC-IDRI-0903', 'LIDC-IDRI-0974',
            'LIDC-IDRI-0026', 'LIDC-IDRI-0122', 'LIDC-IDRI-0193', 'LIDC-IDRI-0265', 'LIDC-IDRI-0336', 'LIDC-IDRI-0407', 'LIDC-IDRI-0478', 'LIDC-IDRI-0620', 'LIDC-IDRI-0691', 'LIDC-IDRI-0762', 'LIDC-IDRI-0833', 'LIDC-IDRI-0904', 'LIDC-IDRI-0975',
            'LIDC-IDRI-0027', 'LIDC-IDRI-0123', 'LIDC-IDRI-0194', 'LIDC-IDRI-0266', 'LIDC-IDRI-0337', 
            #'LIDC-IDRI-0408',  #problem with data (length data not as expected)
            'LIDC-IDRI-0479', 'LIDC-IDRI-0621', 'LIDC-IDRI-0692', 'LIDC-IDRI-0763', 'LIDC-IDRI-0834', 'LIDC-IDRI-0905', 'LIDC-IDRI-0976',
            'LIDC-IDRI-0028', 'LIDC-IDRI-0124', 'LIDC-IDRI-0195', 'LIDC-IDRI-0267', 'LIDC-IDRI-0338', 'LIDC-IDRI-0409', 'LIDC-IDRI-0480', 'LIDC-IDRI-0622', 'LIDC-IDRI-0693', 'LIDC-IDRI-0764', 'LIDC-IDRI-0835', 'LIDC-IDRI-0906', 'LIDC-IDRI-0977',
            'LIDC-IDRI-0029', 'LIDC-IDRI-0125', 'LIDC-IDRI-0196', 'LIDC-IDRI-0268', 'LIDC-IDRI-0339', 'LIDC-IDRI-0410', 'LIDC-IDRI-0551', 'LIDC-IDRI-0623', 'LIDC-IDRI-0694', 'LIDC-IDRI-0765', 'LIDC-IDRI-0836', 'LIDC-IDRI-0907', 'LIDC-IDRI-0978',
            'LIDC-IDRI-0030', 'LIDC-IDRI-0126', 'LIDC-IDRI-0197', 'LIDC-IDRI-0269', 'LIDC-IDRI-0340', 'LIDC-IDRI-0411', 'LIDC-IDRI-0552', 'LIDC-IDRI-0624', 'LIDC-IDRI-0695', 'LIDC-IDRI-0766', 'LIDC-IDRI-0837', 'LIDC-IDRI-0908', 'LIDC-IDRI-0979',
            'LIDC-IDRI-0031', 'LIDC-IDRI-0127', 'LIDC-IDRI-0198', 'LIDC-IDRI-0270', 'LIDC-IDRI-0341', 'LIDC-IDRI-0412', 'LIDC-IDRI-0553', 'LIDC-IDRI-0625', 'LIDC-IDRI-0696', 'LIDC-IDRI-0767', 'LIDC-IDRI-0838', 'LIDC-IDRI-0909', 'LIDC-IDRI-0980',
            'LIDC-IDRI-0032', 'LIDC-IDRI-0128', 'LIDC-IDRI-0199', 'LIDC-IDRI-0271', 'LIDC-IDRI-0342', 'LIDC-IDRI-0413', 'LIDC-IDRI-0554', 'LIDC-IDRI-0626', 'LIDC-IDRI-0697', 'LIDC-IDRI-0768', 'LIDC-IDRI-0839', 'LIDC-IDRI-0910', 'LIDC-IDRI-0981',
            'LIDC-IDRI-0033', 'LIDC-IDRI-0129', 'LIDC-IDRI-0200', 'LIDC-IDRI-0272', 'LIDC-IDRI-0343', 'LIDC-IDRI-0414', 'LIDC-IDRI-0555', 'LIDC-IDRI-0627', 'LIDC-IDRI-0698', 'LIDC-IDRI-0769', 'LIDC-IDRI-0840', 'LIDC-IDRI-0911', 'LIDC-IDRI-0982',
            'LIDC-IDRI-0034', 'LIDC-IDRI-0130', 'LIDC-IDRI-0201', 'LIDC-IDRI-0273', 'LIDC-IDRI-0344', 'LIDC-IDRI-0415', 'LIDC-IDRI-0556', 'LIDC-IDRI-0628', 'LIDC-IDRI-0699', 'LIDC-IDRI-0770', 'LIDC-IDRI-0841', 'LIDC-IDRI-0912', 'LIDC-IDRI-0983',
            'LIDC-IDRI-0035', 'LIDC-IDRI-0131', 'LIDC-IDRI-0202', 'LIDC-IDRI-0274', 'LIDC-IDRI-0345', 'LIDC-IDRI-0416', 'LIDC-IDRI-0557', 'LIDC-IDRI-0629', 'LIDC-IDRI-0700', 'LIDC-IDRI-0771', 'LIDC-IDRI-0842', 'LIDC-IDRI-0913', 'LIDC-IDRI-0984',
            'LIDC-IDRI-0036', 'LIDC-IDRI-0132', 'LIDC-IDRI-0203', 'LIDC-IDRI-0275', 'LIDC-IDRI-0346', 'LIDC-IDRI-0417', 'LIDC-IDRI-0558', 'LIDC-IDRI-0630', 'LIDC-IDRI-0701', 'LIDC-IDRI-0772', 'LIDC-IDRI-0843', 'LIDC-IDRI-0914', 'LIDC-IDRI-0985',
            'LIDC-IDRI-0037', 'LIDC-IDRI-0133', 'LIDC-IDRI-0204', 'LIDC-IDRI-0276', 'LIDC-IDRI-0347', 'LIDC-IDRI-0418', 'LIDC-IDRI-0559', 'LIDC-IDRI-0631', 'LIDC-IDRI-0702', 'LIDC-IDRI-0773', 'LIDC-IDRI-0844', 'LIDC-IDRI-0915', 'LIDC-IDRI-0986',
            'LIDC-IDRI-0038', 'LIDC-IDRI-0134', 'LIDC-IDRI-0205', 'LIDC-IDRI-0277', 'LIDC-IDRI-0348', 'LIDC-IDRI-0419', 'LIDC-IDRI-0560', 'LIDC-IDRI-0632', 'LIDC-IDRI-0703', 'LIDC-IDRI-0774', 'LIDC-IDRI-0845', 'LIDC-IDRI-0916', 'LIDC-IDRI-0987',
            'LIDC-IDRI-0039', 'LIDC-IDRI-0135', 'LIDC-IDRI-0206', 'LIDC-IDRI-0278', 'LIDC-IDRI-0349', 'LIDC-IDRI-0420', 'LIDC-IDRI-0561', 'LIDC-IDRI-0633', 'LIDC-IDRI-0704', 'LIDC-IDRI-0775', 'LIDC-IDRI-0846', 'LIDC-IDRI-0917', 'LIDC-IDRI-0988',
            'LIDC-IDRI-0040', 'LIDC-IDRI-0136', 'LIDC-IDRI-0207', 'LIDC-IDRI-0279', 'LIDC-IDRI-0350', 'LIDC-IDRI-0421', 'LIDC-IDRI-0562', 'LIDC-IDRI-0634', 'LIDC-IDRI-0705', 'LIDC-IDRI-0776', 'LIDC-IDRI-0847', 'LIDC-IDRI-0918', 'LIDC-IDRI-0989',
            'LIDC-IDRI-0041', 'LIDC-IDRI-0137', 'LIDC-IDRI-0208', 'LIDC-IDRI-0280', 'LIDC-IDRI-0351', 'LIDC-IDRI-0422', 'LIDC-IDRI-0563', 'LIDC-IDRI-0635', 'LIDC-IDRI-0706', 'LIDC-IDRI-0777', 'LIDC-IDRI-0848', 'LIDC-IDRI-0919', 'LIDC-IDRI-0990',
            'LIDC-IDRI-0042', 'LIDC-IDRI-0138', 'LIDC-IDRI-0209', 'LIDC-IDRI-0281', 'LIDC-IDRI-0352', 'LIDC-IDRI-0423', 'LIDC-IDRI-0564', 'LIDC-IDRI-0636', 'LIDC-IDRI-0707', 'LIDC-IDRI-0778', 'LIDC-IDRI-0849', 'LIDC-IDRI-0920', 'LIDC-IDRI-0991',
            'LIDC-IDRI-0043', 'LIDC-IDRI-0139', 'LIDC-IDRI-0210', 'LIDC-IDRI-0282', 'LIDC-IDRI-0353', 'LIDC-IDRI-0424', 'LIDC-IDRI-0565', 'LIDC-IDRI-0637', 'LIDC-IDRI-0708', 'LIDC-IDRI-0779', 'LIDC-IDRI-0850', 'LIDC-IDRI-0921', 'LIDC-IDRI-0992',
            'LIDC-IDRI-0044', 'LIDC-IDRI-0140', 'LIDC-IDRI-0211', 'LIDC-IDRI-0283', 'LIDC-IDRI-0354', 'LIDC-IDRI-0425', 'LIDC-IDRI-0566', 'LIDC-IDRI-0638', 'LIDC-IDRI-0709', 'LIDC-IDRI-0780', 'LIDC-IDRI-0851', 'LIDC-IDRI-0922', 'LIDC-IDRI-0993',
            'LIDC-IDRI-0045', 'LIDC-IDRI-0141', 'LIDC-IDRI-0212', 'LIDC-IDRI-0284', 'LIDC-IDRI-0355', 'LIDC-IDRI-0426', 'LIDC-IDRI-0567', 'LIDC-IDRI-0639', 'LIDC-IDRI-0710', 'LIDC-IDRI-0781', 'LIDC-IDRI-0852', 'LIDC-IDRI-0923', 'LIDC-IDRI-0994',
            'LIDC-IDRI-0046', 'LIDC-IDRI-0142', 'LIDC-IDRI-0213', 'LIDC-IDRI-0285', 'LIDC-IDRI-0356', 'LIDC-IDRI-0427', 'LIDC-IDRI-0568', 'LIDC-IDRI-0640', 'LIDC-IDRI-0711', 'LIDC-IDRI-0782', 'LIDC-IDRI-0853', 'LIDC-IDRI-0924', 'LIDC-IDRI-0995',
            'LIDC-IDRI-0047', 'LIDC-IDRI-0143', 'LIDC-IDRI-0214', 'LIDC-IDRI-0286', 'LIDC-IDRI-0357', 'LIDC-IDRI-0428', 'LIDC-IDRI-0569', 'LIDC-IDRI-0641', 'LIDC-IDRI-0712', 'LIDC-IDRI-0783', 'LIDC-IDRI-0854', 'LIDC-IDRI-0925', 'LIDC-IDRI-0996',
            'LIDC-IDRI-0048', 'LIDC-IDRI-0144', 'LIDC-IDRI-0215', 'LIDC-IDRI-0287', 'LIDC-IDRI-0358', 'LIDC-IDRI-0429', 'LIDC-IDRI-0570', 'LIDC-IDRI-0642', 'LIDC-IDRI-0713', 'LIDC-IDRI-0784', 'LIDC-IDRI-0855', 'LIDC-IDRI-0926', 'LIDC-IDRI-0997',
            'LIDC-IDRI-0049', 'LIDC-IDRI-0145', 'LIDC-IDRI-0216', 'LIDC-IDRI-0288', 'LIDC-IDRI-0359', 'LIDC-IDRI-0430', 'LIDC-IDRI-0571', 'LIDC-IDRI-0643', 'LIDC-IDRI-0714', 'LIDC-IDRI-0785', 'LIDC-IDRI-0856', 'LIDC-IDRI-0927', 'LIDC-IDRI-0998',
            'LIDC-IDRI-0050', 'LIDC-IDRI-0146', 'LIDC-IDRI-0217', 'LIDC-IDRI-0289', 'LIDC-IDRI-0360', 'LIDC-IDRI-0431', 'LIDC-IDRI-0572', 'LIDC-IDRI-0644', 'LIDC-IDRI-0715', 'LIDC-IDRI-0786', 'LIDC-IDRI-0857', 'LIDC-IDRI-0928', 'LIDC-IDRI-0999',
            'LIDC-IDRI-0076', 'LIDC-IDRI-0147', 'LIDC-IDRI-0218', 'LIDC-IDRI-0290', 'LIDC-IDRI-0361', 'LIDC-IDRI-0432', 'LIDC-IDRI-0573', 'LIDC-IDRI-0645', 'LIDC-IDRI-0716', 'LIDC-IDRI-0787', 'LIDC-IDRI-0858', 'LIDC-IDRI-0929', 'LIDC-IDRI-1000',
            'LIDC-IDRI-0077', 'LIDC-IDRI-0148', 'LIDC-IDRI-0219', 'LIDC-IDRI-0291', 'LIDC-IDRI-0362', 'LIDC-IDRI-0433', 'LIDC-IDRI-0574', 'LIDC-IDRI-0646', 'LIDC-IDRI-0717', 'LIDC-IDRI-0788', 'LIDC-IDRI-0859', 'LIDC-IDRI-0930', 'LIDC-IDRI-1001',
            'LIDC-IDRI-0078', 'LIDC-IDRI-0149', 'LIDC-IDRI-0220', 'LIDC-IDRI-0292', 'LIDC-IDRI-0363', 'LIDC-IDRI-0434', 'LIDC-IDRI-0575', 'LIDC-IDRI-0647', 'LIDC-IDRI-0718', 'LIDC-IDRI-0789', 'LIDC-IDRI-0860', 'LIDC-IDRI-0931', 'LIDC-IDRI-1002',
            'LIDC-IDRI-0079', 'LIDC-IDRI-0150', 'LIDC-IDRI-0221', 'LIDC-IDRI-0293', 'LIDC-IDRI-0364', 'LIDC-IDRI-0435', 'LIDC-IDRI-0576', 'LIDC-IDRI-0648', 'LIDC-IDRI-0719', 'LIDC-IDRI-0790', 'LIDC-IDRI-0861', 'LIDC-IDRI-0932', 'LIDC-IDRI-1003',
            'LIDC-IDRI-0080', 'LIDC-IDRI-0151', 'LIDC-IDRI-0222', 'LIDC-IDRI-0294', 'LIDC-IDRI-0365', 'LIDC-IDRI-0436', 'LIDC-IDRI-0577', 'LIDC-IDRI-0649', 'LIDC-IDRI-0720', 'LIDC-IDRI-0791', 'LIDC-IDRI-0862', 'LIDC-IDRI-0933', 'LIDC-IDRI-1004',
            'LIDC-IDRI-0081', 'LIDC-IDRI-0152', 'LIDC-IDRI-0223', 'LIDC-IDRI-0295', 'LIDC-IDRI-0366', 'LIDC-IDRI-0437', 'LIDC-IDRI-0578', 'LIDC-IDRI-0650', 'LIDC-IDRI-0721', 'LIDC-IDRI-0792', 'LIDC-IDRI-0863', 'LIDC-IDRI-0934', 'LIDC-IDRI-1005',
            'LIDC-IDRI-0082', 'LIDC-IDRI-0153', 'LIDC-IDRI-0224', 'LIDC-IDRI-0296', 'LIDC-IDRI-0367', 'LIDC-IDRI-0438', 'LIDC-IDRI-0579', 'LIDC-IDRI-0651', 'LIDC-IDRI-0722', 'LIDC-IDRI-0793', 'LIDC-IDRI-0864', 'LIDC-IDRI-0935', 'LIDC-IDRI-1006',
            'LIDC-IDRI-0083', 'LIDC-IDRI-0154', 'LIDC-IDRI-0225', 'LIDC-IDRI-0297', 'LIDC-IDRI-0368', 'LIDC-IDRI-0439', 'LIDC-IDRI-0580', 'LIDC-IDRI-0652', 'LIDC-IDRI-0723', 'LIDC-IDRI-0794', 'LIDC-IDRI-0865', 'LIDC-IDRI-0936', 'LIDC-IDRI-1007',
            'LIDC-IDRI-0084', 'LIDC-IDRI-0155', 'LIDC-IDRI-0226', 'LIDC-IDRI-0298', 'LIDC-IDRI-0369', 'LIDC-IDRI-0440', 'LIDC-IDRI-0581', 'LIDC-IDRI-0653', 'LIDC-IDRI-0724', 'LIDC-IDRI-0795', 'LIDC-IDRI-0866', 'LIDC-IDRI-0937', 'LIDC-IDRI-1008',
            'LIDC-IDRI-0085', 'LIDC-IDRI-0156', 'LIDC-IDRI-0227', 'LIDC-IDRI-0299', 'LIDC-IDRI-0370', 'LIDC-IDRI-0441', 'LIDC-IDRI-0582', 'LIDC-IDRI-0654', 'LIDC-IDRI-0725', 'LIDC-IDRI-0796', 'LIDC-IDRI-0867', 'LIDC-IDRI-0938', 'LIDC-IDRI-1009',
            'LIDC-IDRI-0086', 'LIDC-IDRI-0157', 'LIDC-IDRI-0228', 'LIDC-IDRI-0300', 'LIDC-IDRI-0371', 'LIDC-IDRI-0442', 'LIDC-IDRI-0583', 'LIDC-IDRI-0655', 'LIDC-IDRI-0726', 'LIDC-IDRI-0797', 'LIDC-IDRI-0868', 'LIDC-IDRI-0939', 'LIDC-IDRI-1010',
            'LIDC-IDRI-0087', 'LIDC-IDRI-0158', 'LIDC-IDRI-0229', 'LIDC-IDRI-0301', 'LIDC-IDRI-0372', 'LIDC-IDRI-0443', 'LIDC-IDRI-0584', 'LIDC-IDRI-0656', 'LIDC-IDRI-0727', 'LIDC-IDRI-0798', 'LIDC-IDRI-0869', 'LIDC-IDRI-0940', 'LIDC-IDRI-1011',
            'LIDC-IDRI-0088', 'LIDC-IDRI-0159', 'LIDC-IDRI-0230', 'LIDC-IDRI-0302', 'LIDC-IDRI-0373', 'LIDC-IDRI-0444', 'LIDC-IDRI-0586', 'LIDC-IDRI-0657', 'LIDC-IDRI-0728', 'LIDC-IDRI-0799', 'LIDC-IDRI-0870', 'LIDC-IDRI-0941', 'LIDC-IDRI-1012',
            'LIDC-IDRI-0089', 'LIDC-IDRI-0160', 'LIDC-IDRI-0231', 'LIDC-IDRI-0303', 'LIDC-IDRI-0374', 'LIDC-IDRI-0445', 'LIDC-IDRI-0587', 'LIDC-IDRI-0658', 'LIDC-IDRI-0729', 'LIDC-IDRI-0800', 'LIDC-IDRI-0871', 'LIDC-IDRI-0942',
            'LIDC-IDRI-0090', 'LIDC-IDRI-0161', 'LIDC-IDRI-0232', 'LIDC-IDRI-0304', 'LIDC-IDRI-0375', 'LIDC-IDRI-0446', 'LIDC-IDRI-0588', 'LIDC-IDRI-0659', 'LIDC-IDRI-0730', 'LIDC-IDRI-0801', 'LIDC-IDRI-0872', 'LIDC-IDRI-0943',
            'LIDC-IDRI-0091', 'LIDC-IDRI-0162', 'LIDC-IDRI-0233', 'LIDC-IDRI-0305', 'LIDC-IDRI-0376', 'LIDC-IDRI-0447', 'LIDC-IDRI-0589', 'LIDC-IDRI-0660', 'LIDC-IDRI-0731', 'LIDC-IDRI-0802', 'LIDC-IDRI-0873', 'LIDC-IDRI-0944',
            'LIDC-IDRI-0092', 'LIDC-IDRI-0163', 'LIDC-IDRI-0234', 'LIDC-IDRI-0306', 'LIDC-IDRI-0377', 'LIDC-IDRI-0448', 'LIDC-IDRI-0590', 'LIDC-IDRI-0661', 'LIDC-IDRI-0732', 'LIDC-IDRI-0803', 'LIDC-IDRI-0874', 'LIDC-IDRI-0945',
            'LIDC-IDRI-0093', 'LIDC-IDRI-0164', 'LIDC-IDRI-0235', 'LIDC-IDRI-0307', 'LIDC-IDRI-0378', 'LIDC-IDRI-0449', 'LIDC-IDRI-0591', 'LIDC-IDRI-0662', 'LIDC-IDRI-0733', 'LIDC-IDRI-0804', 'LIDC-IDRI-0875', 'LIDC-IDRI-0946',
            'LIDC-IDRI-0094', 'LIDC-IDRI-0165',            
            #'LIDC-IDRI-0236',  #problem with data (length data not as expected)
            'LIDC-IDRI-0308', 'LIDC-IDRI-0379', 'LIDC-IDRI-0450', 'LIDC-IDRI-0592', 'LIDC-IDRI-0663', 'LIDC-IDRI-0734', 'LIDC-IDRI-0805', 'LIDC-IDRI-0876', 'LIDC-IDRI-0947',
            'LIDC-IDRI-0095', 'LIDC-IDRI-0166', 'LIDC-IDRI-0237', 'LIDC-IDRI-0309', 'LIDC-IDRI-0380', 'LIDC-IDRI-0451', 'LIDC-IDRI-0593', 'LIDC-IDRI-0664', 'LIDC-IDRI-0735', 'LIDC-IDRI-0806', 'LIDC-IDRI-0877', 'LIDC-IDRI-0948',
            'LIDC-IDRI-0096', 'LIDC-IDRI-0167', 'LIDC-IDRI-0239', 'LIDC-IDRI-0310', 'LIDC-IDRI-0381', 'LIDC-IDRI-0452', 'LIDC-IDRI-0594', 'LIDC-IDRI-0665', 'LIDC-IDRI-0736', 'LIDC-IDRI-0807', 'LIDC-IDRI-0878', 'LIDC-IDRI-0949'
            ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu" ### for height = 1024 need to test gpu with big mem
print(f"Found torch device is {device}")
#
if torch.cuda.device_count() > 1 :
    print("Using", torch.cuda.device_count(), "GPUs!")
# transform for image resizing
transform_resize = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(size=height), transforms.ToTensor() ]) 
#
# folder to save volumes
if not os.path.isdir(output_folder_volume):
    os.makedirs(output_folder_volume)

for idx_patient,pid in enumerate(patient_list):
    idx_patient += pid_offset
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first() # scan class instance for patient ID
    print(f"scan for patient {pid} is {scan}")

    #ann = pl.query(pl.Annotation).filter(pl.Scan.patient_id == pid).first()
    #ann.visualize_in_3d(edgecolor='green', cmap='autumn')
    
    # %%
    vol = scan.to_volume().astype(np.float32) # creating numpy array with scan volume. dtype to match type of example volume in diffdrr library
    #vol = ann.scan.to_volume().astype(np.float32)
    #print(scan)
    #print(scan.pixel_spacing)
    ## Save scan in nii.gz file 
    #print(vol.shape)
    spacing_CT = np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing])
    vol_itk=vol.copy()
    vol_itk=np.moveaxis(vol_itk, -1,  0) ## to make compativle when going trrough itk
    #print(vol_itk.shape)

    vol_itk=itk.image_view_from_array(vol_itk)
    #vol_itk.SetSpacing([spacing_CT[2],spacing_CT[1],spacing_CT[0]])
    vol_itk.SetSpacing(spacing_CT)
    filename_volume=os.path.join(output_folder_volume, f'Patient{idx_patient:04d}.mhd')
    itk.imwrite(vol_itk,filename_volume, True)
    

    
    print(f'volume saved as {filename_volume}')
    #
    vol_shape=vol.shape
    print(vol_shape)

    if visualize_on:
        ann=scan.annotations ## annotation in scan
    

        i,j,k = ann[0].centroid
        print(f'centroid {i}, {j}, {k}')

        

        plt.imshow(vol[:,:,int(k)], cmap=plt.cm.gray)
        plt.plot(j, i, '.r', label="Nodule centroid")
        plt.legend()
        plt.show()

    
    #
    #print(vol_shape)
    #print(type(vol))
    #

    # ### Compute x-ray projections and create Annotations Gold Standard
    # 
    # Compute x-ray projections from CT volumes. Also, create a volume with the annotations.

    # %%    
    ### parameters of the geometry of projection
    # We use diff DRR made to have differentiable DRR from CT volumes. Diff DRR is customized for cone-beam CTs
    # https://github.com/eigenvivek/DiffDRR
    #    
    
    print(f"CT voxel resolution is {spacing_CT} mm")
    # intrinsics cam #
    # Initialize the DRR module for generating synthetic X-rays
    with torch.no_grad():
        torch.cuda.empty_cache()
    #
    
    #
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

    #print(translation)

    # %%
    #| cuda
    ## create drr object from CT
    drr = DRR(
        vol,      # The CT volume as a numpy array
        spacing_CT,     # Voxel dimensions of the CT
        sdr=sdr,   # Source-to-detector radius (half of the source-to-detector distance)
        height=drr_height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=delx_drr,    # Pixel spacing (in mm)
    )
    # paralelization
    if torch.cuda.device_count() > 1:
        drr = nn.DataParallel(drr)
    drr = drr.to(device)
    drr.eval()
    

    #device="cpu"

    # %%
    # Create folder 3D labels 3D
    if not os.path.isdir(output_folder_labels3d):
        os.mkdir(output_folder_labels3d)

    # %%
    
    #print(os.path.join(folder_cam_patient,f'{idx_patient:02d}.txt'))
    #np.savetxt( os.path.join(folder_cam_patient,f'{0:02d}.txt'), K.ravel(),delimiter=',')


    # %%
    #| cuda
    ##########################################
    ### loop over patients and create data ###
    ##########################################
    #
    ### create drr images using drr object for N_views. Each view is a rotation around the x axis of 360°/N_views from 0°
    # create list with params cam
    #filename_params_cam = os.path.join(output_folder, f'paramas_cam.txt')
    #
    # create lists with params
    #
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
    #
    ############################################
    # create folders to store data per patient #
    ############################################
    # folder output calibs per patient
    folder_cam_patient=os.path.join(output_folder_cams, f'Patient{idx_patient:04d}/')
    if not os.path.isdir(folder_cam_patient):
        os.makedirs(folder_cam_patient)
    #
    # folder output images per patient
    folder_image_patient=os.path.join(output_folder_images, f'Patient{idx_patient:04d}/')
    if not os.path.isdir(folder_image_patient):
        os.makedirs(folder_image_patient)
    #
    # folder output 2D labels
    folder_labels2d_patient=os.path.join(output_folder_labels2d, f'Patient{idx_patient:04d}/')
    if not os.path.isdir(folder_labels2d_patient):
        os.makedirs(folder_labels2d_patient)

            
    #
    for idx_view in range(N_views):
        #
        np.savetxt( os.path.join(folder_cam_patient,f'{idx_patient:04d}_{idx_view:02d}.txt'), K.ravel(),
                    delimiter=',',
                    newline=','
                ) ###
        
        rad_x = idx_view*2*torch.pi/N_views
        #
        rotation = torch.tensor([[rad_x, rot_y , rot_z]], device=device) 
        q = utils.convert(rotation=rotation, input_parameterization='euler_angles', input_convention='ZYX', output_parameterization="quaternion")
        #print(q)
        params_ext_cam.append(q)
        #
        # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
        # For example, quaternions, rotation matrix, axis-angle, etc...
        with torch.no_grad():
            print(f'rotation {rotation}')
            print(f'translation {translation}')
            img = drr(rotation=rotation, translation=translation, parameterization="euler_angles", convention="ZYX")
        
                 
        filename_image=os.path.join(folder_image_patient, f'Image_{idx_view:02d}.png')

        im_norm=img.squeeze().cpu().detach()
        im_norm = (im_norm-im_norm.min())/(im_norm.max()-im_norm.min())        

        # resample to output size    
        #transform = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(size=height), transforms.ToTensor() ]) 
        #im_norm = [transform(x_) for x_ in im_norm] 
        im_norm = transform_resize(im_norm)

        #save image
        save_image(im_norm.squeeze().cpu().detach(), filename_image)

        del img
        
        
    del drr

    # %%
    #| cuda
    #Imagenes, bb3D, bb2D para un paciente.
    #device='cpu'
    ann=scan.annotations ## annotation in scan
    print(f"number of annotations for patient {pid} are {len(ann)}")
    ### Loop over annotations for CT volumes
    bb_2d=[[] for n in range(N_views)] ## list with 2D bounding boxes for all views
    bb_3d=[] 
    #
    #a=ann[1]
    #idx_ann=1
 
    for idx_ann, a in enumerate(ann):
        #a=ann[1]
        #idx_ann=1
        #  
        #print(f"annotation {idx_ann}")
        #print(a)
        #print(a.bbox())
        ## create volume for the annotation
        vol_ann = np.zeros(vol.shape, dtype=np.float32)
        #print(type(vol_ann[0,0,0]))
        #print(type(vol[0,0,0]))
        #mask_ann = a.boolean_mask()
        #
        #vol_ann[a.bbox()][a.boolean_mask()] = 1  ### a.bbox() contains the gold standard (3D bounding box) of the localisation of each annotation 
        vol_ann[a.bbox()] = 200
        #print(len(np.argwhere(vol_ann > 0)))
        
        #print(f"number of voxels in the 3D bbox is {len(np.argwhere(vol > 0))}")
        #print(vol_ann.max())
        #
        bb_3d.append(a.bbox_matrix())
        print(f'bb 3D is {bb_3d}')
        ##  vol_ann is a binary volume the same size as the CT images, where all pixels are zero outside the box containing the indications and 1 inside the box
        #
        ## create drr object for the volume of the anotation
        ## create drr object from CT
        drr_ann = DRR(
                vol_ann,   # The binary volume with the annotations
                #vol,
                spacing_CT,   # Voxel dimensions of the volume
                sdr=sdr,   # Source-to-detector radius (half of the source-to-detector distance)
                height=drr_height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
                delx=delx_drr,    # Pixel spacing (in mm)
            )

        if torch.cuda.device_count() > 1:
            drr_ann = nn.DataParallel(drr_ann)
        drr_ann = drr_ann.to(device)
        drr_ann.eval()

        #print(drr_ann) 
        for idx_view in range(N_views):  
            #
            ## generates the xray images 
            #
            rad_x = idx_view*2*torch.pi/N_views
            #print(rad_x)
            rotation = torch.tensor([[rad_x, rot_y , rot_z]], device=device) 
            #
            
            # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
            # For example, quaternions, rotation matrix, axis-angle, etc...
            with torch.no_grad():
                img = drr_ann(rotation=rotation, translation=translation, parameterization="euler_angles", convention="ZYX")
            
            # resample to output size    
            #transform = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(size=height), transforms.ToTensor() ]) 
            #img = [transform(x_) for x_ in img] 
            img = transform_resize(img.squeeze().cpu().detach())
            
            # get pixels inside
            img[img>0] = 1
            #
            ### here use your favorite tool to save the created xray images s.
            ##visualisation
            #       
            nonzero_indices = np.argwhere(img.squeeze().cpu().detach().numpy().squeeze() > 0)
            #print(nonzero_indices)
            if len(nonzero_indices) > 0:        
                #print(nonzero_indices)
                #
                # Calculate bounding box coordinates
                min_row, min_cln = np.min(nonzero_indices, axis=0)
                max_row, max_cln = np.max(nonzero_indices, axis=0)
                #
                # Bounding box dimensions
                width_bb = max_row - min_row + 1
                height_bb = max_cln - min_cln + 1
                # save bounding_box
                bb_2d[idx_view].append([min_row, min_cln, width_bb, height_bb, 'nodule',a.subtlety, a.internalStructure, a.calcification, a.sphericity, a.margin, a.lobulation, a.spiculation, a.texture, a.malignancy])
                print(f'Non-empty projection for the 3D bounding box of annotation {idx_ann} on view {idx_view}')

                if visualize_on:
                    fig, ax = plt.subplots()
                    ax.imshow(img.squeeze())
                    rect = patches.Rectangle((min_cln, min_row), height_bb, width_bb, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.show()
                    print(f'view {idx_view+1} : {bb_2d[idx_view]}')


            else:
                print(f'Empty projection for the 3D bounding box of annotation {idx_ann} on view {idx_view}')
            
            del img
        del vol_ann
        del drr_ann
        del rotation
        
        print('drr_ann deleted')

    # write bounding boxes
    for idx_view, bb_2d_view in enumerate(bb_2d):
        if len(bb_2d_view):
            f=open(os.path.join(folder_labels2d_patient,f'Cam_{idx_view:02d}_bbox2d.txt'), 'w')
            for l_bb2d in bb_2d_view:
                print(l_bb2d)
                f.writelines([f'{elem},' for elem in l_bb2d]+['\n'])
            f.close()
    #
    # write bounding box 3D
    #print(bb_3d)
    # TODO : check geom drr for  bounding box to center of the geometry
    c_vol = np.array(vol.shape)/2. # center voxel
    c_vol *= np.array(spacing_CT) # coordinates center voxel in 
    print(f'center CT {c_vol}')
    # TODO : to check if origin at center of voxel of at a vertex
    if len(bb_3d):
        bb_3d_nuscenes=np.array(bb_3d, dtype=type(spacing_CT[0]))
        #print(bb_3d_nuscenes.shape)
        bb_3d_nuscenes[:,0,:] *= spacing_CT[0]
        bb_3d_nuscenes[:,1,:] *= spacing_CT[1]
        bb_3d_nuscenes[:,2,:] *= spacing_CT[2]
        bb_3d_nuscenes[:,:,1] = bb_3d_nuscenes[:,:,1]- bb_3d_nuscenes[:,:,0] ### size faces of bb3D
        """
        print(f'bb_3d_nuscenes not centered is {bb_3d_nuscenes}')
        np.savetxt(os.path.join(output_folder_labels3d,f'Patient_{idx_patient:04d}_bbox3d_nonCentered.txt'),
                bb_3d_nuscenes.reshape(bb_3d_nuscenes.shape[0],6),
                delimiter=',',
                newline='\n')
        """
        bb_3d_nuscenes[:,:,0] = bb_3d_nuscenes[:,:,0] - c_vol # pos with regards to center of volume
        print(f'bb_3d_nuscenes centered is {bb_3d_nuscenes}')
        #print(bb_3d_nuscenes.reshape(bb_3d_nuscenes.shape[0],6))
        np.savetxt(os.path.join(output_folder_labels3d,f'Patient_{idx_patient:04d}_bbox3d.txt'),
                bb_3d_nuscenes.reshape(bb_3d_nuscenes.shape[0],6),
                delimiter=',',
                newline='\n')
        print(f'3d bounding box saved for Patient {idx_patient:04d}')
    else:
        print(f'no annotation detected for patient {idx_patient:02d}')
    
    # add patient to list of patients processed
    with open(filename_patient_processed, 'a') as file:
        file.write(f'{pid}, {idx_patient}\n')
    
    print(f'Patient {pid} with ID nu_scenes {idx_patient}  processed')
