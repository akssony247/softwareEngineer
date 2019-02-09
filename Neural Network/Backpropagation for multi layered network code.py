# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:56:10 2018

@author: Akshay
"""

# repeat of Test_list

import numpy as np
import math
import matplotlib.pyplot as plt
import time

A=input('Please enter the number of neurons in each layer')
A=list(map(int, A.split(',')))
data = np.genfromtxt(r'<Your path>MnistData.txt')  #import the data file
dataResult=np.genfromtxt(r'<Your path>MnistLabels.txt')
dataResult=np.ascontiguousarray(dataResult, dtype=np.int)
dataResult_hot = np.zeros((5000, 10))
dataResult_hot[np.arange(5000),dataResult] = 1

dataConcat=np.concatenate((data,dataResult_hot),axis=1)
TrainData=dataConcat[0:4000,:]
TrainResult=dataConcat[4000:5000,:]

Weight_1=np.random.rand(A[0],784)
Weight_1=Weight_1
Weight_2=np.random.rand(10,A[-1])
Weight_2=Weight_2
Weight_3=np.zeros_like(Weight_2)
Weight_4=np.zeros_like(Weight_1)
Eta=0.05
Eta2=0.03
Eta1=0.15
Eta4=0.005
Moment=0.1
epoch=2
Repetitions=1
error_rate_Train=list()
error_rate_Test=list()
#create a list of arrays
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
for delta in range(len(A)-1):
    dell.append(np.zeros(A[delta]))
    

#=======================================================================================
def forwardfirst(Input_matrix,Layer):
    for ff in range(A[Layer+1]):
        ListOfHiddenSum[Layer][ff]=np.sum(Input_matrix*(ListOfMatrices[Layer][ff])*Eta2)
    ListOfHiddenSumSig[Layer]=[(1/(1+math.exp(-x))) for x in ListOfHiddenSum[Layer]]
    Layer=Layer+1
    if Layer<(len(A)-1):
        forwardfirst(ListOfHiddenSumSig[Layer-1],Layer)
#    else:
#        return ListOfHiddenSumSig[Layer-1]
#========================================================================================
def backwardfirst(delli_matrix,Layer):
    for t in range(A[Layer-1]):
        for p in range(A[Layer-2]):
            if Layer>=3:
                ListOfMatricesChange[Layer-2][t][p]=Eta1*delli_matrix[t]*(ListOfHiddenSumSig[Layer-2][t])*(1-((ListOfHiddenSumSig[Layer-2][t])))*(ListOfHiddenSumSig[Layer-3][p])
            else:ListOfMatricesChange[Layer-2][t][p]=Eta1*delli_matrix[t]*(ListOfHiddenSumSig[Layer-2][t])*(1-((ListOfHiddenSumSig[Layer-2][t])))*(H1[p])
    ListOfMatrices[Layer-2]=ListOfMatrices[Layer-2]+ListOfMatricesChange[Layer-2]
    for l in range(A[Layer-2]):
        for k in range(A[Layer-1]):
            dell[Layer-2][l]=dell[Layer-2][l]+delli_matrix[k]*(ListOfHiddenSumSig[Layer-2][k])*(1-(ListOfHiddenSumSig[Layer-2][k]))*Eta*(ListOfMatrices[Layer-2][k][l])
    Layer=Layer-1
    if Layer>=2:
        backwardfirst(dell[Layer-1],Layer)
        
#=======================================================================================
#Forward loop 
for g in range(1):
    for h in range(epoch):
        print(epoch)
        np.random.shuffle(TrainData)
        dataTrainData=TrainData[:,0:784]
        dataTrainResult=TrainData[:,784:794]
        for i in range(4000):
            print(i)
        
            S1=np.zeros(A[0])
            for a1 in range(A[0]):
                S1[a1]=np.sum(dataTrainData[i]*Weight_1[a1]*Eta4*Eta)
            H1=[(1/(1+math.exp(-x))) for x in S1]
            Layer=0
            Hfinal=forwardfirst(H1,Layer)
            Si=np.zeros(10)
            yCap=np.zeros(10)
            for b1 in range(10):
                Si[b1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[b1]*Eta2)
            yCap=[(1/(1+math.exp(-x))) for x in Si]
            ei=dataTrainResult[i]-yCap
            
         #Reverse loop
            for j in range(10):
                for k in range(A[-1]):
                    Weight_3[j,k]=(Weight_3[j,k]*Moment)+(ei[j])*(yCap[j])*(1-(yCap[j]))*(ListOfHiddenSumSig[len(A)-2][k])*Eta
                Weight_2=Weight_2+Weight_3
            delli=np.zeros(A[-1])
            for ll in range(A[-1]):
                for n in range(10):
                    delli[ll]=delli[ll]+((ei[n])*(yCap[n])*(1-(yCap[n]))*Weight_2[n,ll])
            Layer=len(A)
            backwardfirst(delli,Layer)
            for m in range(A[0]):
                for d in range(784):
                    Weight_4[m,d]=(Weight_4[m,d]*Moment)+((Eta1)*(H1[m])*(1-(H1[m]))*dell[0][m]*dataTrainData[i,d])
            Weight_1=Weight_1+Weight_4
#Trainerrors==================================================================================        
        correct=0
        Layer=0
        print('trainerror')
        for o in range(4000):
            S12=np.zeros(A[0])
            for c1 in range(A[0]):
                S12[c1]=np.sum(dataTrainData[o]*Weight_1[c1]*Eta1)
            H12=[(1/(1+math.exp(-x))) for x in S12]
            Hfinal=forwardfirst(H12,Layer)
            Si2=np.zeros(10)
            for d1 in range(10):
                Si2[d1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[d1]*Eta2)
            yCap2=[(1/(1+math.exp(-x))) for x in Si2]
            determined_value=np.argmax(yCap2, axis=0)
            actual_value=np.argmax(dataTrainResult[o], axis=0)
            if determined_value==actual_value:
                correct=correct+1
        print(correct)
        error_rate_Train.append(1-(correct/4000))
#Trainerrors==================================================================================                
        TrainResultData=TrainResult[:,0:784]
        TrainResultResult=TrainResult[:,784:794]
        correct=0
        Layer=0
        print('testerror')
        for o in range(1000):
            S12=np.zeros(A[0])
            for c1 in range(A[0]):
                S12[c1]=np.sum(TrainResultData[o]*Weight_1[c1]*Eta4)
            H12=[(1/(1+math.exp(-x))) for x in S12]
            Hfinal=forwardfirst(H12,Layer)
            Si2=np.zeros(10)
            for d1 in range(10):
                Si2[d1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[d1]*Eta2)
            yCap2=[(1/(1+math.exp(-x))) for x in Si2]
            determined_value=np.argmax(yCap2, axis=0)
            actual_value=dataResult[o+4000]
            
            if determined_value==actual_value:
                correct=correct+1
        print(correct)
        error_rate_Test.append(1-(correct/1000))
#confusion matrix for Training data=======================================================
ConfusionMatTrain=np.zeros((10,10))
print('con1')
for p in range(4000):
    Layer=0
    S13=np.zeros(A[0])
    for j1 in range(A[0]):
        S13[j1]=np.sum(dataTrainData[p]*Weight_1[j1]*Eta4)
    
    H13=[(1/(1+math.exp(-x))) for x in S13]
    Hfinal=forwardfirst(H13,Layer)
    Si3=np.zeros(10)
    for k1 in range(10):
        Si3[k1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[k1]*Eta2)
    yCap2=[(1/(1+math.exp(-x))) for x in Si3]
    determined_value=np.argmax(yCap2, axis=0)
    actual_value=dataResult[p]
    ConfusionMatTrain[determined_value,actual_value]=ConfusionMatTrain[determined_value,actual_value]+1       
#confusion matrix for Training data=======================================================
#confusion matrix for Testing data=======================================================
print('con2')
ConfusionMatTest=np.zeros((10,10))
for p in range(1000):
    Layer=0
    S13=np.zeros(A[0])
    for j1 in range(A[0]):
        S13[j1]=np.sum(TrainResultData[p]*Weight_1[j1]*Eta4)
    H13=[(1/(1+math.exp(-x))) for x in S13]
    Hfinal=forwardfirst(H13,Layer)
    Si3=np.zeros(10)
    for k1 in range(10):
        Si3[k1]=np.sum(ListOfHiddenSumSig[len(A)-2]*Weight_2[k1]*Eta2)
    yCap2=[(1/(1+math.exp(-x))) for x in Si3]
    determined_value=np.argmax(yCap2, axis=0)
    actual_value=dataResult[4000+p]
    ConfusionMatTest[determined_value,actual_value]=ConfusionMatTest[determined_value,actual_value]+1
#confusion matrix for Testing data=======================================================
#graph for both error rates.


timecalculate=timecalculate-time.time()
