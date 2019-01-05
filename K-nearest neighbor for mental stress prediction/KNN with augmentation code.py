# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 00:56:39 2018

@author: Anirudh
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from matplotlib.colors import ListedColormap
#import data-Please change the path to your local machines file location.
data = np.genfromtxt(r'<YOur path>ISKNN.csv',delimiter=',')
Row,Col=data.shape
data1=np.zeros((Row,2))
IndTrials=10

#initiate the parameter lists to collect mean performance metrics for each neighbor value.
Mean_R_hitrate=list()
Mean_R_Specificity=list()
Mean_R_Sensitivity=list()
Mean_R_PPV=list()
Mean_R_NPV=list()
Dev_R_hitrate=list()
Dev_R_Sensitivity=list()
Dev_R_Specificity=list()
Dev_R_PPV=list()
Dev_R_NPV=list()

Mean_R=list()
Standard_D=list()
for Radius in range(5,15):# iterate for different values of Radius.
    Radius=Radius*0.1
    hitrate=list()
    Sensitivity=list()
    Specificity=list()
    PPV=list()
    NPV=list()
    for z in range(IndTrials):
        data2=np.concatenate((data, data1), axis = 1) # add an empty matrix of 2 columns to the data matrix.
        np.random.shuffle(data2)
        dataTrain=data2[:320,:]
        dataTest=data2[320:,:]
        for i in range(79):# calculate distance of all points with each point in dataset.
            for j in range(320):
                  dataTrain[j,3]=(((dataTest[i,0]-dataTrain[j,0])**2)+((dataTest[i,1]-dataTrain[j,1])**2))**(1/2)
             # store the distances in a column and sort based on that column so that all the lower distances comes initially.
            data3=dataTrain[dataTrain[:,3].argsort()] 
            data4=np.array(list(filter(lambda x: x[3] < Radius, data3))) #slice the matrix till only where the distances less than radius.
            x=0
            x1=0
            y=0
            y1=0
            #check in all neighbors for their classes within the specified radius and .
            for z in range(0,int(data4.size/5)):
                if data4[z,2]==1:
                    x1=x1+1
                else:
                    y1=y1+1
            if x1==0:
                dataTest[i,4]=0
            elif y1==0:
                dataTest[i,4]=1
            else:
                if (x1)<=(y1):
                    dataTest[i,4]=1
                else:
                    dataTest[i,4]=0
        falsePositive=0
        TruePositive=0
        falseNegative=0
        TrueNegative=0
        #calculate the true positives,true negatives, false positives and false negatives.
        for p in range(79):
            if dataTest[p,2]==1:
                
                if dataTest[p,2]==dataTest[p,4]:
                    TruePositive=TruePositive+1
                else:
                    falseNegative=falseNegative+1
            else:
                if dataTest[p,2]==dataTest[p,4]:
                    TrueNegative=TrueNegative+1
                else:
                    falsePositive=falsePositive+1
        
        hitrate.append((TruePositive+TrueNegative)/79)
        Sensitivity.append(TruePositive/(TruePositive+falseNegative))
        Specificity.append(TrueNegative/(TrueNegative+falsePositive))
        PPV.append(TruePositive/(TruePositive+falsePositive))
        NPV.append(TrueNegative/(TrueNegative+falseNegative))
    # calculate the mean of the performance parameters for particular radius value for 10 independent iterations.
    Mean_R_hitrate.append(st.mean(hitrate))
    Mean_R_Sensitivity.append(st.mean(Sensitivity))
    Mean_R_Specificity.append(st.mean(PPV))
    Mean_R_PPV.append(st.mean(PPV))
    Mean_R_NPV.append(st.mean(Specificity))
    Dev_R_hitrate.append(st.stdev(hitrate))
    Dev_R_Sensitivity.append(st.stdev(Sensitivity))
    Dev_R_Specificity.append(st.stdev(PPV))
    Dev_R_PPV.append(st.stdev(PPV))
    Dev_R_NPV.append(st.stdev(Specificity))
radiusTest=[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]        
# plot errorbar garphs for performance parameters over mean values and deviations.
plt.figure()
plt.errorbar(radiusTest,Mean_R_hitrate,Dev_R_hitrate)
plt.xlabel('Radius')
plt.ylabel('Mean hitrate')
plt.title('Mean hitrate vs Radius')
plt.show()

plt.figure()
plt.errorbar(radiusTest,Mean_R_Sensitivity,Dev_R_Sensitivity)
plt.xlabel('Radius')
plt.ylabel('Mean Sensitivity')
plt.title('Mean Sensitivity vs Radius')
plt.show()

plt.figure()
plt.errorbar(radiusTest,Mean_R_Specificity,Dev_R_Specificity)
plt.xlabel('Radius')
plt.ylabel('Mean Specificity')
plt.title('Mean Specificity vs Radius')
plt.show()

plt.figure()
plt.errorbar(radiusTest,Mean_R_PPV,Dev_R_PPV)
plt.xlabel('Radius')
plt.ylabel('Mean PPV')
plt.title('Mean PPV vs Radius')
plt.show()

plt.figure()
plt.errorbar(radiusTest,Mean_R_NPV,Dev_R_NPV)
plt.xlabel('Radius')
plt.ylabel('Mean NPV')
plt.title('Mean NPV vs Radius')
plt.show()
# from the above metrics we get that accuracy is highest for radius=0.8
# following code draws a meshgrid and plots decision boundary.
gx=np.linspace(0,14,141)
gy=np.linspace(0,25,251)
xx,yy=np.meshgrid(gx,gy)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#function to determine the class of a point on meshgrid.
def model1(element1,element2):
    for k in range(320):
        dataTrain[k,3]=(((element1-dataTrain[k,0])**2)+((element2-dataTrain[k,1])**2))**(1/2)
        data3=dataTrain[dataTrain[:,3].argsort()]
        data4=np.array(list(filter(lambda x: x[3] < 0.6, data3)))
    x=0
    x1=0
    y=0
    y1=0
    #check in all neighbors which are in specified radius.
    for z in range(0,int(data4.size/5)):
        if data4[z,2]==1:
            x1=x1+1
            
        else:
            y1=y1+1
        if x1==0 and y1==0:
            return -1
        elif x1==0:
            return 0
        elif y1==0:
            return 1
        else:
            if (x1<=y1):
                return 0
            else:
                return 1
# apply model1 function to each point on meshgrid and  store the values in Z matrix
Z=np.zeros_like(xx)
for out in range(141):
    for inn in range(251):
        Z[inn,out]=model1(xx[inn,out],yy[inn,out])
#plot the meshgrid, the contour, and scatter of the datapoints.   
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(dataTrain[:,0],dataTrain[:,1], dataTrain[:,2],cmap=cmap_bold)
plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(dataTrain[:, 0], dataTrain[:, 1], c=dataTrain[:,2])
plt.xlabel('Wife Salary')
plt.ylabel('Husband Salary')
plt.title('Decision boundary for Radius KNN')








