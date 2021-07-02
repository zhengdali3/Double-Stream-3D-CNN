#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Model
import os
import numpy as np
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


# In[ ]:


def validate(LOSOCV, filepathdir):
    
    f = open(filepathdir + 'result', 'w')
    
    print(type(f))
    
    uf1_list = []
    acc_list = []
    uar_list = []
    max_acc = np.zeros((30, 3))
    weight_path = filepathdir
    
    weightlisting = os.listdir(filepathdir)
    
    if(len(LOSOCV[0]) == 6):
        model = Model.DS()
    elif(len(LOSOCV[0]) == 4):
        model = Model.origin()
    elif(len(LOSOCV[0]) == 8):
        model = Model.DS_domain()
    
    for weight in weightlisting:
        
        if(weight.split('-')[0] == 'weights'):
            model.load_weights(weight_path + weight)

            i = int(weight.split('-')[-1][:-5])

            if(len(LOSOCV[0]) == 6):
                y_pred1 = model.predict([LOSOCV[i][3], LOSOCV[i][4]])
                y_pred = np.argmax(y_pred1, axis=1)
                y_test = np.argmax(LOSOCV[i][5], axis=1)

            elif(len(LOSOCV[0]) == 4):
                y_pred1 = model.predict(LOSOCV[i][2])
                y_pred = np.argmax(y_pred1, axis=1)
                y_test = np.argmax(LOSOCV[i][3], axis=1)

            elif(len(LOSOCV[0]) == 8):
                y_pred1 = model.predict([LOSOCV[i][4], LOSOCV[i][5]])
                y_pred = np.argmax(y_pred1[0], axis=1)
                y_test = np.argmax(LOSOCV[i][6], axis=1)            

            accuracy = accuracy_score(y_test, y_pred)
            if(accuracy > max_acc[i][0]):
                matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                FP = matrix.sum(axis=0) - np.diag(matrix)
                FN = matrix.sum(axis=1) - np.diag(matrix)
                TP = np.diag(matrix)
                TN = matrix.sum() - (FP + FN + TP)
                Nc = matrix.sum(axis = 1)
                c = 0

                f1_s = np.ones([3])
                uar = 0
                deno = (2 * TP + FP + FN)
                for j in range(3):
                    if deno[j] != 0:
                        f1_s[j] = (2 * TP[j]) / (2 * TP[j] + FP[j] + FN[j])
                    else:
                        f1_s[j] = 1

                    if Nc[j] !=0:
                        uar_s = TP[j] / Nc[j]
                        uar = uar + uar_s
                        c = c + 1

                uar = uar / c
                f1 = np.mean(f1_s)


                print('Sample ', i, ': ', LOSOCV[i][0].shape,LOSOCV[i][1].shape ,LOSOCV[i][2].shape,LOSOCV[i][3].shape)


                print('Individual f1: ', f1)
                max_acc[i][1] = f1

                print('Individual UAR: ', uar)
                max_acc[i][2] = uar

                print('Individual accuracy: ' , accuracy)
                max_acc[i][0] = accuracy

    for a in range(30):
        if(max_acc[a][0] != 0):
            f.write('Sample '+ str(a) + ':\n')
            f.write('Individual accuracy: ' + str(max_acc[a][0]) + '\n')
            f.write('Individual UF1: ' + str(max_acc[a][1]) + '\n')
            f.write('Individual UAR: ' + str(max_acc[a][2]) + '\n')
            acc_list.append(max_acc[a][0])
            uf1_list.append(max_acc[a][1])
            uar_list.append(max_acc[a][2])
    
    f.write('Accuracy with LOSOCV is :' + str(statistics.mean(acc_list)) + 'UF1: ' + str(statistics.mean(uf1_list)) +  'UAR: ' + str(statistics.mean(uar_list)) + '\n')
    f.close()
    print('Accuracy with LOSOCV is :' , statistics.mean(acc_list), 'UF1: ', statistics.mean(uf1_list), 'UAR: ', statistics.mean(uar_list))

