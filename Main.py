#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Model
import Training
import Prepare_TIM as Prepare
import Validation
import tensorflow as tf


# In[ ]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[2]:


SAMM_dataset_path = '../SAMM/'
SMIC_dataset_path = '../SMIC/SMIC_all_cropped/HS/'
CK_dataset_path = '../CK+/'
weightspathdir = 'weights_SAMM_FA/'

model = 'DS'
MM = True
OF = True
fps = 100
#SAMM = Prepare.prepareSAMM(MM, OF, fps, SAMM_dataset_path)
#CK = Prepare.prepareCK(MM, OF, fps, CK_dataset_path)
#SMIC = Prepare.prepareSMIC(MM, OF, fps, SMIC_dataset_path)
LOSOCV = Training.getLOSOCV('DS', 'SAMM', 'CK+')
Training.training(LOSOCV, weightspathdir)
Validation.validate(LOSOCV, weightspathdir)


# In[ ]:




