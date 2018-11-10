# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:58:24 2018

@author: Anirudh
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import data-Please change the path to your local machines file location.
data0 = np.genfromtxt(r'<your path>ISKNN.csv',delimiter=',')
Row,Col=data0.shape
data1=np.zeros((Row,1))
data=np.concatenate((data0,data1),axis=1) #add an empty matrix of 2 rows to data matrix.
np.random.shuffle(data)
trainingData=data[0:320,:]
testingData=data[300:,:]
w0=1
w1=1
w2=1
ErrorRate=list()
eta=0.1
Epoch=25
Repetitions=10
for l in range(Repetitions):  #iteration of set of epochs for multiple times.  
    for i in range(Epoch): # iteration of algorithm over the dataset for epoch times.
        for j in range(320):
            f=float(trainingData[j,0])*w1+float(trainingData[j,1])*w2+w0
            if (f<0.001 and f>-0.001):
                break
            else:
                if f>=0:
                    k=1
                else:k=0
                trainingData[j,3]=k
                error=k-float(trainingData[j,2])
                w0=w0-eta*error
                w1=w1-eta*error*trainingData[j,0]
                w2=w2-eta*error*trainingData[j,1]
    TruePositives=0
    TrueNegatives=0
    FalsePositives=0
    FalseNegatives=0
    for t in range(99):
        testingData1=float(testingData[t,0])*w1+float(testingData[t,1])*w2+w0
        if testingData1<=0:
            testingData[t,3]=0
        else: testingData[t,3]=1
    dataFinal=np.concatenate((trainingData,testingData),axis=0)
    for j in range(399):
        if dataFinal[j,2]==0 and dataFinal[j,3]==0:
            TrueNegatives=TrueNegatives+1
        elif (dataFinal[j,2]==1) and (dataFinal[j,3]==0):
            FalseNegatives=FalseNegatives+1
        elif (dataFinal[j,2]==1) and (dataFinal[j,3]==1):
            TruePositives=TruePositives+1
        else:FalsePositives=FalsePositives+1
    ErrorRate.append(1-((TruePositives+TrueNegatives)/399))
     
TPtrain=0
TNtrain=0
FPtrain=0
FNtrain=0
for z in range(300):
    if (trainingData[z,2]==0) and (trainingData[z,3]==0):
        TNtrain=TNtrain+1
    elif (trainingData[z,2]==1) and (trainingData[z,3]==0):
        FNtrain=FNtrain+1
    elif (trainingData[z,2]==1) and (trainingData[z,3]==1):
        TPtrain=TPtrain+1
    else:FPtrain=FPtrain+1
#performance parameters over training data
HitRate1=(TPtrain+TNtrain)/300
Sensitivity1=TPtrain/(TPtrain+FNtrain)
Specificity1=TNtrain/(TNtrain+FPtrain)
PPV1=TPtrain/(TPtrain+FPtrain)
NPV1=TNtrain/(TNtrain+FNtrain)

TPtrain1=0
TNtrain1=0
FPtrain1=0
FNtrain1=0

for z1 in range(100):
    if (trainingData[z1,2]==0) and (trainingData[z1,3]==0):
        TNtrain1=TNtrain1+1
    elif (trainingData[z1,2]==1) and (trainingData[z1,3]==0):
        FNtrain1=FNtrain1+1
    elif (trainingData[z1,2]==1) and (trainingData[z1,3]==1):
        TPtrain1=TPtrain1+1
    else:FPtrain1=FPtrain1+1
#performance parameters over
HitRate=(TPtrain1+TNtrain1)/99
Sensitivity=TPtrain1/(TPtrain1+FNtrain1)
Specificity=TNtrain1/(TNtrain1+FPtrain1)
PPV=TPtrain1/(TPtrain1+FPtrain1)
NPV=TNtrain1/(TNtrain1+FNtrain1)
#plot error graph
plt.plot(range(0,500,50),ErrorRate)
plt.xlabel('Epoch')
plt.ylabel('ErrorRate')
plt.title('Perceptron Error for different epoch')
plt.show()
# plot bar graph of performance parameters on training data
objects1 = ('HitRate1', 'Sensitivity', 'Specificity', 'PPV', 'NPV')
y_pos1 = np.arange(len(objects1))
performance1 = [HitRate1,Sensitivity1,Specificity1,PPV1,NPV1]
plt.bar(y_pos1, performance1, align='center', alpha=0.5)
plt.xticks(y_pos1, objects1)
plt.ylabel('Performance-Testing')
plt.title('Performance Parameters-Testing')
plt.show()

# plot bar graph of performance parameters on training data
objects = ('HitRate', 'Sensitivity', 'Specificity', 'PPV', 'NPV')
y_pos = np.arange(len(objects))
performance = [HitRate,Sensitivity,Specificity,PPV,NPV]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance-Training')
plt.title('Performance Parameters-Training')
plt.show()

# code to plot the decision boundary.
gx=np.linspace(0,14,141)
gy=np.linspace(0,25,251)
xx,yy=np.meshgrid(gx,gy)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#function which decides whether a point on meshgrid belongs to class 1 or class 2.
def model1(element1,element2):
    Threshhold=element1*w1+element2*w2+w0
    if Threshhold>0:
        return 1
    else: return 0
# apply model1 function to all points and save the output in matrix Z. 
Z=np.zeros_like(xx)
for out in range(141):
    for inn in range(251):
        Z[inn,out]=model1(xx[inn,out],yy[inn,out])
#plot the meshgrid, the contour, and scatter of the datapoints.    
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(trainingData[:,0],trainingData[:,1], trainingData[:,2],cmap=cmap_bold)
plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(trainingData[:, 0], trainingData[:, 1], c=trainingData[:,2])
plt.xlabel('Wife Salary')
plt.ylabel('Husband Salary')
plt.title('Decision boundary for perceptron')












     








