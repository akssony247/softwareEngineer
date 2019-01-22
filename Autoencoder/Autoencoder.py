# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:51:00 2019

@author: aksso
"""

#===python Code Starts========import Libraries===================================================
import numpy as np
import math
import matplotlib.pyplot as plt
import time
#===============================Take input and dataset=====================================
A=input('Please enter the number of neurons in each layer')
A=list(map(int, A.split(',')))
data = np.genfromtxt(r'DATASET_PATH')  #import the data file
dataResult=np.genfromtxt(r'LABEL PATH')
dataResult=np.ascontiguousarray(dataResult, dtype=np.int)
dataResult_hot = np.zeros((5000, 10))
dataResult_hot[np.arange(5000),dataResult] = 1
dataConcat=np.concatenate((data,dataResult_hot),axis=1)
TrainData=dataConcat[0:4000,:]
TrainResult=dataConcat[4000:5000,:]
Weight_1=np.random.rand(A[0],784)
Weight_1=Weight_1
Weight_2=np.random.rand(784,A[-1])
Weight_2=Weight_2
Weight_3=np.zeros_like(Weight_2)
Weight_4=np.zeros_like(Weight_1)
Eta=0.05
Eta2=0.03
Eta1=0.15
Eta4=0.005
Moment=0.1
epoch=10
Repetitions=5
error_rate_Train=list()
error_rate_Test=list()
#create a list of arrays
#===================initialize weight matrices for hidden layers=====================
timecalculate=time.time()
ListOfMatrices=list()
ListOfMatricesChange=list()
ListOfHiddenSum=list()
ListOfHiddenSumSig=list()
dell=list()
for hidden in range(len(A)-1):
    ListOfMatrices.append(np.random.rand(A[hidden+1],A[hidden]))
for hiddenChange in range(len(A)-1):
    ListOfMatricesChange.append(np.zeros((A[hiddenChange+1],A[hiddenChange])))
for hiddenSum in range(1,len(A)):
    ListOfHiddenSum.append(np.zeros(A[hiddenSum]))
for hiddenSumSig in range(1,len(A)):
    ListOfHiddenSumSig.append(np.zeros(A[hiddenSumSig]))
#====================Function for activating the hidden neurons========================
def forwardfirst(Input_matrix,Layer):
    for ff in range(A[Layer+1]):
        ListOfHiddenSum[Layer][ff]=np.sum(Input_matrix*(ListOfMatrices[Layer][ff])*Eta2)
    ListOfHiddenSumSig[Layer]=[(1/(1+math.exp(-x))) for x in ListOfHiddenSum[Layer]]
    Layer=Layer+1
    if Layer<(len(A)-1):
        forwardfirst(ListOfHiddenSumSig[Layer-1],Layer)
#    else:
#        return ListOfHiddenSumSig[Layer-1]
#==Function for backpropogating the errors as dell matrix and changing the weight matrices==
def backwardfirst(delli_matrix,Layer):
    for t in range(A[Layer-1]):
        for p in range(A[Layer-2]):
            if Layer>=3:
                ListOfMatricesChange[Layer-2][t][p]=Eta1*delli_matrix[t]*(ListOfHiddenSumSig[Layer-2][t])*(1-((ListOfHiddenSumSig[Layer-2][t])))*(ListOfHiddenSumSig[Layer-3][p])
            else:ListOfMatricesChange[Layer-2][t][p]=Eta1*delli_matrix[t]*(ListOfHiddenSumSig[Layer-2][t])*(1-((ListOfHiddenSumSig[Layer-2][t])))*(H1[p])
    ListOfMatrices[Layer-2]=ListOfMatrices[Layer-2]+ListOfMatricesChange[Layer-2]
    for l in range(A[Layer-2]):
        for k in range(A[Layer-1]):
            dell[Layer-2][l]=dell[Layer-2][l]+delli_matrix[k]*(ListOfHiddenSumSig[Layer-2][k])*(1-(ListOfHiddenSumSig[Layer-2][k]))*(ListOfMatrices[Layer-2][k][l])
    Layer=Layer-1
    if Layer>=2:
        backwardfirst(dell[Layer-1],Layer)
#=================================================
#Forward loop 
for g in range(Repetitions):
    for h in range(epoch):
        print(epoch)
        np.random.shuffle(TrainData)
        dataTrainData=TrainData[:,0:784]
        dataTrainResult=TrainData[:,784:794]
        CostFunctionTrain=list()
        for i in range(4000):
            print(i)
            S1=np.zeros(A[0])
            for a1 in range(A[0]):
                S1[a1]=np.sum(dataTrainData[i]*Weight_1[a1]*Eta4*Eta)
            H1=[(1/(1+math.exp(-x))) for x in S1]
            Layer=0
            Si=np.zeros(784)
            yCap=np.zeros(784)
            if (len(A)>1):
                Hfinal=forwardfirst(H1,Layer)
                for b1 in range(784):
                    Si[b1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[b1]*Eta2)
            else:
                for b1 in range(784):
                    Si[b1]=np.sum(H1*Weight_2*Eta2)
                
            yCap=[(1/(1+math.exp(-x))) for x in Si]
            ei=dataTrainData[i]-yCap
            CostFunction=[err**2 for err in ei]
            CostFunctionTrain.append(np.sum(CostFunction)
         #Reverse loop
            if np.sum(CostFunctionTrain)<=2:
                break
            else:
                for j in range(784):
                    for k in range(A[-1]):
                        Weight_3[j,k]=(Weight_3[j,k]*Moment)+(ei[j])*(yCap[j])*(1-(yCap[j]))*(ListOfHiddenSumSig[len(A)-2][k])*Eta
                    Weight_2=Weight_2+Weight_3
                delli=np.zeros(A[-1])
                for ll in range(A[-1]):
                    for n in range(784):
                        delli[ll]=delli[ll]+((ei[n])*(yCap[n])*(1-(yCap[n]))*Weight_2[n,ll])
                Layer=len(A)
                dell=list()
                if (len(A)>1):
                    for delta in range(len(A)-1):
                        dell.append(np.zeros(A[delta]))
                    backwardfirst(delli,Layer)
                for m in range(A[0]):
                    for d in range(784):
                        Weight_4[m,d]=(Weight_4[m,d]*Moment)+((Eta1)*(H1[m])*(1-(H1[m]))*dell[0][m]*dataTrainData[i,d])
                Weight_1=Weight_1+Weight_4
            if np.sum(CostFunctionTrain)<=2:
                break
#==========================Plot the features (start)===========================
fig=plt.figure(figsize=(8, 8))
columns = 10
rows = 10
for a1 in range(100):
    img=np.reshape(Weight_2[:,a1], (28,28),order='F')
    fig.add_subplot(rows, columns, a1+1)
    plt.imshow(img, cmap="gray")
plt.show() 

#==========================Plot the features (end)=============================
#==========================plot the bar graph digit wise(start)===================

dataResult=np.genfromtxt(r'C:\Users\Anirudh\Desktop\Intelligent systems\MnistLabels.txt')
dataResult=np.ascontiguousarray(dataResult, dtype=np.int)

dataResult_hot = np.zeros((5000, 10))
dataResult_hot[np.arange(5000),dataResult] = 1


dataConcat=np.concatenate((data,dataResult_hot),axis=1)
dataConcatTrain=dataConcat[0:4000,:]
dataConcatTest=dataConcat[4000:5000,:]

CostTrain=list()
for hotcode in range(10):
    temp=dataConcatTrain[dataConcatTrain[:,784+hotcode]>0]
    rows,cols=np.shape(temp)
    CostFunctionTrain=list()
    for training in range(rows):
        S1=np.zeros(100)
        for a1 in range(100):
            S1[a1]=np.sum(temp[training,0:784]*Weight_1[a1])
        H1=[(1/(1+math.exp(-x))) for x in S1]
        Si=np.zeros(784)
        yCap=np.zeros(784)
        for b1 in range(784):
            Si[b1]=np.sum(H1*Weight_2[b1])
        yCap=[(1/(1+math.exp(-x))) for x in Si]
        ei=temp[training,0:784]-yCap
        CostFunction=[err**2 for err in ei]
        CostFunctionTrain.append(np.sum(CostFunction))
    CostTrain.append(np.sum(CostFunctionTrain)/rows)
CostTest=list()
for hotcode in range(10):
    temp=dataConcatTest[dataConcatTest[:,784+hotcode]>0]
    rows,cols=np.shape(temp)
    CostFunctionTest=list()
    for training in range(rows):
        S1=np.zeros(100)
        for a1 in range(100):
            S1[a1]=np.sum(temp[training,0:784]*Weight_1[a1])
        H1=[(1/(1+math.exp(-x))) for x in S1]
        Si=np.zeros(784)
        yCap=np.zeros(784)
        for b1 in range(784):
            Si[b1]=np.sum(H1*Weight_2[b1])
        yCap=[(1/(1+math.exp(-x))) for x in Si]
        ei=temp[training,0:784]-yCap
        CostFunction=[err**2 for err in ei]
        CostFunctionTest.append(np.sum(CostFunction))
    CostTest.append(np.sum(CostFunctionTrain)/rows)
n_groups = 10
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, CostTrain, bar_width,
                 alpha=opacity,
                 color='b',
                 label='CostTrain')
rects2 = plt.bar(index + bar_width, CostTest, bar_width,
                 alpha=opacity,
                 color='g',
                 label='CostTest')
plt.xlabel('Digits')
plt.ylabel('Cost Function Values')
plt.title('Accuracy for different digits')
plt.xticks(index + bar_width, (0,1,2,3,4,5,6,7,8,9))
plt.legend()
plt.tight_layout()
plt.show()
#==========================plot the bar graph digit wise(end)=======================

#==========================Plot the error graph on different epochs=================
Epochs=[1,2,3,4,5,6,7,8,9,10]
plt.plot(Epochs,CostFunctionTrain)
plt.title('CostFunction vs Epochs')
plt.ylabel('Cost Function')
plt.xlabel('Epochs')
plt.show()