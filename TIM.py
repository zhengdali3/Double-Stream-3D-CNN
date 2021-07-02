#!/usr/bin/env python

import torch
import math

import cv2
import numpy
import numpy as np

import softsplat

""
assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

""
def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()
    # end

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])
# end

# +
# ""
backwarp_tenGrid = {}
# -

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

""
# def interpolation(first, second)

# +
# tenFirst = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/first.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
# tenSecond = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/second.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
# tenFlow = torch.FloatTensor(numpy.ascontiguousarray(read_flo('./images/flow.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()

# +
# tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

# +
# for intTime, fltTime in enumerate(numpy.linspace(0.0, 1.0, 11).tolist()):
# 	tenSummation = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=None, strType='summation')
# 	tenAverage = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=None, strType='average')
# 	tenLinear = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=(0.3 - tenMetric).clip(0.0000001, 1.0), strType='linear') # finding a good linearly metric is difficult, and it is not invariant to translations
# 	tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter

# 	cv2.imshow(winname='summation', mat=tenSummation[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
# 	cv2.imshow(winname='average', mat=tenAverage[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
# 	cv2.imshow(winname='linear', mat=tftenLinear[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
# 	cv2.imshow(winname='softmax', mat=tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
# 	cv2.waitKey(delay=0)
# # return winname='softmax', mat=tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)

# +
# print(numpy.linspace(0.0, 1.0, 3).tolist())
# -

def TIM(selected_frames, expanded_n):
    
    depth = selected_frames.shape[0]
    
    
    ori_frames = []
    
    result = numpy.zeros((expanded_n,selected_frames.shape[1],selected_frames.shape[2],3),dtype='float')
    
    for i in range(depth):
        ori_frames.append(torch.FloatTensor(numpy.ascontiguousarray(selected_frames[i].transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda())
    
    print("ori_frame shape:", ori_frames[0][0, :, :, :].cpu().numpy().transpose(1, 2, 0).shape)
                          
    if(depth < expanded_n):
        
        int_base = math.floor((expanded_n - depth) / (depth - 1))
        int_residue = (expanded_n - depth) % (depth - 1)
        j = 0
        ori_index = 0
        pr_f = False
        pr_s = False
                          
        while(j < expanded_n):
            
            result[j,: ,: , :] = ori_frames[ori_index][0,:,:,:].cpu().numpy().transpose(1, 2, 0)
            j = j + 1
            
            if(j == expanded_n):
                break
                          
            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.float32(selected_frames[ori_index]), cv2.COLOR_BGR2GRAY), cv2.cvtColor(np.float32(selected_frames[ori_index + 1]), cv2.COLOR_BGR2GRAY), flow=None,
                                                            pyr_scale=0.5, levels=1, winsize=15,
                                                            iterations=2,
                                                            poly_n=5, poly_sigma=1.1, flags=0).transpose(2,0,1)[None, :, :, :]
            if(not pr_f):
                pr_f = True
              
            tenFirst = ori_frames[ori_index]
            tenSecond = ori_frames[ori_index+1]
            tenFlow = torch.FloatTensor(numpy.ascontiguousarray(flow)).cuda()
            
            tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)
            
            if(ori_index < int_residue):
                interFrameNum = int(int_base + 1)
            else:
                interFrameNum = int(int_base)
            
            for intTime, fltTime in enumerate(numpy.linspace(0.0, 1.0, interFrameNum + 2).tolist()):
                if(fltTime != 0.0 and fltTime != 1.0):
                    tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax')
                    if(not pr_s):
                        pr_s = True
                    result[j, :, :, :] = tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
                    j = j + 1
            
            ori_index = ori_index + 1
    
    else:
        for i in range(expanded_n):
            result[i,: ,: , :] = ori_frames[i][0, :, :, :].cpu().numpy().transpose(1, 2, 0)           
    return result

