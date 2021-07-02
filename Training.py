#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import Model
from keras.callbacks import ModelCheckpoint

image_rows = 64
image_columns = 64
image_depth = 18

flow_rows = 144
flow_columns = 120
flow_depth = 16


# In[2]:


def getLOSOCV(model, Dataset, CK):
    if(isinstance(Dataset, list)):
        training_set, labels, subject_boundary, flow_set = Dataset[0], Dataset[1], Dataset[2], Dataset[3]
        sample = Dataset[0].shape[0]
    else:
        training_set, labels, subject_boundary, flow_set = np.load(Dataset+ '/' + Dataset + '_set_TIM_FA.npy' ), np.load(Dataset+ '/' + Dataset + '_labels.npy' ), np.load(Dataset+ '/' + Dataset + '_subject_boundary.npy'), np.load(Dataset+ '/' + Dataset + '_flow.npy')
        sample = training_set.shape[0]
        
    if(isinstance(CK, list) and CK):
        CK_training_set, CK_labels, CK_flow = CK[0], CK[1], CK[2]
        CK_num = 327
    else:
        if(isinstance(CK, list) and not CK):
            CK_num = 0
        elif (CK == 'CK+'):
            CK_num = 327
            CK_training_set, CK_labels, CK_flow = np.load("CK+/CK_set_TIM_FA.npy") , np.load("CK+/CK_labels.npy"), np.load('CK+/CK_flow.npy')
    
    LOSOCV = []

    
    print("Sample size: ", sample)
    
    for i in range(len(subject_boundary) - 1):    
        # 0 is for micro-expression dataset and 1 is for CK+ dataset

        training_images = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 1, image_rows, image_columns, image_depth))
        training_labels = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 3))
        training_domain_labels = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num))
        training_flow_images = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 1, flow_rows, flow_columns, flow_depth))

        verification_images = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 1, image_rows, image_columns, image_depth))
        verification_labels = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 3))
        verification_domain_labels = np.zeros((subject_boundary[i + 1] - subject_boundary[i]))
        verification_flow_images = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 1, flow_rows, flow_columns, flow_depth))

        for j in range(sample):
            if (j >= subject_boundary[i] and j < subject_boundary[i + 1]):
                verification_images[j-subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
                verification_labels[j-subject_boundary[i]][:] = labels[j][:]
                verification_flow_images[j-subject_boundary[i]][0][:][:][:] = flow_set[j][0][:][:][:]
            elif (j < subject_boundary[i]):
                training_images[j][0][:][:][:] = training_set[j][0][:][:][:]
                training_labels[j][:] = labels[j][:]
                training_flow_images[j][0][:][:][:] = flow_set[j][0][:][:][:]
            else:
                training_images[j - subject_boundary[i+1] + subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
                training_labels[j - subject_boundary[i+1] + subject_boundary[i]][:] = labels[j][:]
                training_flow_images[j - subject_boundary[i+1] + subject_boundary[i]][0][:][:][:] = flow_set[j][0][:][:][:]

        if(CK == 'CK+' or (isinstance(CK, list) and CK)):
            for k in range(327):
                training_images[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][0][:][:][:] = CK_training_set[k][0][:][:][:]
                training_labels[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][:] = CK_labels[k][:]
                training_domain_labels[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k] = 1
                training_flow_images[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][0][:][:][:] = CK_flow[k][0][:][:][:]

        if(model == 'DS'):
            LOSOCV.append([training_images, training_flow_images, training_labels, verification_images, verification_flow_images, verification_labels])
        elif(model == 'DS_domain'):
            LOSOCV.append([training_images, training_flow_images, training_labels, training_domain_labels, verification_images, verification_flow_images, verification_labels, verification_domain_labels])
        else:
            LOSOCV.append([training_images, training_labels, verification_images, verification_labels])

        
#     if(model == 'DS'):
#         for i in range(len(subject_boundary) - 1):
#             training_images = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 1, image_rows, image_columns, image_depth))
#             training_flow_images = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 1, flow_rows, flow_columns, flow_depth))
#             verification_images = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 1, image_rows, image_columns, image_depth))
#             verification_flow_images = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 1, flow_rows, flow_columns, flow_depth))
#             training_labels = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 3))
#             verification_labels = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 3))
#             for j in range(sample):
#                 if (j >= subject_boundary[i] and j < subject_boundary[i + 1]):
#                     verification_images[j-subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
#                     verification_flow_images[j-subject_boundary[i]][0][:][:][:] = flow_set[j][0][:][:][:]
#                     verification_labels[j-subject_boundary[i]][:] = labels[j][:]
#                 elif (j < subject_boundary[i]):
#                     training_images[j][0][:][:][:] = training_set[j][0][:][:][:]
#                     training_labels[j][:] = labels[j][:]
#                     training_flow_images[j][0][:][:][:] = flow_set[j][0][:][:][:]
#                 else:
#                     training_images[j - subject_boundary[i+1] + subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
#                     training_labels[j - subject_boundary[i+1] + subject_boundary[i]][:] = labels[j][:]
#                     training_flow_images[j - subject_boundary[i+1] + subject_boundary[i]][0][:][:][:] = flow_set[j][0][:][:][:]

#             if(CK == 'CK+' or (isinstance(CK, list) and CK)):
#                 for k in range(327):
#                     training_images[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][0][:][:][:] = CK_training_set[k][0][:][:][:]
#                     training_flow_images[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][0][:][:][:] = CK_flow[k][0][:][:][:]
#                     training_labels[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][:] = CK_labels[k][:]
                    
#             LOSOCV.append([training_images, training_flow_images, training_labels, verification_images, verification_flow_images, verification_labels])
    
#     else:
#         training_images = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 1, image_rows, image_columns, image_depth))
#         verification_images = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 1, image_rows, image_columns, image_depth))
#         training_labels = np.zeros((sample - (subject_boundary[i + 1] - subject_boundary[i]) + CK_num, 3))
#         verification_labels = np.zeros((subject_boundary[i + 1] - subject_boundary[i], 3))
#         for j in range(133):
#             if (j >= subject_boundary[i] and j < subject_boundary[i + 1]):
#                 verification_images[j-subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
#                 verification_labels[j-subject_boundary[i]][:] = labels[j][:]
#             elif (j < subject_boundary[i]):
#                 training_images[j][0][:][:][:] = training_set[j][0][:][:][:]
#                 training_labels[j][:] = labels[j][:]
#             else:
#                 training_images[j - subject_boundary[i+1] + subject_boundary[i]][0][:][:][:] = training_set[j][0][:][:][:]
#                 training_labels[j - subject_boundary[i+1] + subject_boundary[i]][:] = labels[j][:]
               
#         if(CK == 'CK+' or (isinstance(CK, list) and CK)):
#             for k in range(327):
#                 training_images[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][0][:][:][:] = CK_training_set[k][0][:][:][:]
#                 training_labels[sample - (subject_boundary[i + 1] - subject_boundary[i]) + k][:] = CK_labels[k][:]

#         LOSOCV.append([training_images, training_labels, verification_images, verification_labels])
    
    return LOSOCV


# In[4]:


def training(LOSOCV, filepathdir):
    i = 0
    for example in LOSOCV:
        if (len(LOSOCV[0]) == 6):
            if(i == 0):
                model = Model.DS()
                model.save_weights(filepathdir + 'model.h5')
            else:
                model.load_weights(filepathdir + 'model.h5')
            filepath= filepathdir + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}-LOSO-DS-SAMM-" + str(i) +".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            hist = model.fit(x = [example[0], example[1]] ,y = example[2], validation_data = ([example[3], example[4]], example[5]), callbacks=callbacks_list,verbose = 1, batch_size = 8, epochs = 150, shuffle=True)
        elif(len(LOSOCV[0]) == 4):
            if(i == 0):
                model = Model.origin()
                model.save_weights(filepathdir + 'model.h5')
            else:
                model.load_weights(filepathdir + 'model.h5')
            filepath= filepathdir + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}-LOSO-DS-SMIC-" + str(i) +".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            hist = model.fit(example[0], example[1] , validation_data = (example[2], example[3]), callbacks=callbacks_list,verbose = 1, batch_size = 8, epochs = 200, shuffle=True)
        elif(len(LOSOCV[0]) == 8):
            if(i == 0):
                model = Model.DS_domain()
                model.save_weights(filepathdir + 'model.h5')
            else:
                model.load_weights(filepathdir + 'model.h5')
            filepath= filepathdir + "weights-improvement-{epoch:02d}-{val_activation_accuracy:.2f}-LOSO-DS_domain-SAMM-" + str(i) +".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_activation_accuracy', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            hist = model.fit(x = [example[0], example[1]] ,y = [example[2], example[3]], validation_data = ([example[4], example[5]], [example[6], example[7]]), callbacks=callbacks_list,verbose = 1, batch_size = 8, epochs = 150, shuffle=True)

        i = i + 1

