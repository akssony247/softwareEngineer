# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:35:55 2018

@author: Anirudh
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import data-Please change the path to your local machines file location.
data = np.genfromtxt(r'<Your path>ISKNN.csv',delimiter=',')
Row,Col=data.shape
data1=np.zeros((Row,2))
data2=np.concatenate((data, data1), axis = 1) #add an empty matrix of 2 rows to data matrix.
#initialize the performance parameter lists.
HitRate=list()
Sensitivity=list()
Specificity=list()
PPV=list()
NPV=list()
Rang=25
for KNN in range(2,Rang):# iterate for different number of neighbors
    for i in range(Row):# calculate distance of all points with each point in dataset.
        for j in range(Row):
              data2[j,3]=(((data2[j,0]-data2[i,0])**2)+((data2[j,1]-data2[i,1])**2))**(1/2)
         # store the distances in a column and sort based on that column so that all the lower distances comes initially.
        data3=data2[data2[:,3].argsort()]
        x=0
        x1=0
        y=0
        y1=0
    #check in all 'KNN' neighbors and their distances aggregated class wise and mean distance found.
        for z in range(KNN):
            if data3[z,2]==1:
                x1=x1+1
                x=x+data3[z,3]
            else:
                y=y+data3[z,3]
                y1=y1+1
        if x1==0:
            data2[i,4]=0
        elif y1==0:
            data2[i,4]=1
        else:
            if (x/x1)<=(y/y1):
                data2[i,4]=1
            else:
                data2[i,4]=0
    
    falsePositive=0
    TruePositive=0
    falseNegative=0
    TrueNegative=0
            
    #calculate the different parameters of performance
    for p in range(Row):
        if data2[p,2]==1:
            
            if data2[p,2]==data2[p,4]:
                TruePositive=TruePositive+1
            else:
                falseNegative=falseNegative+1
        else:
            if data2[p,2]==data2[p,4]:
                TrueNegative=TrueNegative+1
            else:
                falsePositive=falsePositive+1
    
    HitRate.append((TruePositive+TrueNegative)/Row)
    Sensitivity.append(TruePositive/(TruePositive+falseNegative))
    Specificity.append(TrueNegative/(TrueNegative+falsePositive))
    PPV.append(TruePositive/(TruePositive+falsePositive))
    NPV.append(TrueNegative/(TrueNegative+falseNegative))
# plot bar garph for performance parameters
objects = ('Hitrate', 'Sensitivity', 'Specificity', 'PPV', 'NPV')
y_pos = np.arange(len(objects))
performance = [HitRate[0], Sensitivity[0], Specificity[0], PPV[0], NPV[0]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance')
plt.title('KNN 2 neighbors Augmented performance')
plt.show()
# from the above metrics we get that accuracy is highest for 2 neighbors.
# following code draws a meshgrid and plots decision boundary.
KNN=2
dataTrain=data2[:320,:]
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
    x=0
    x1=0
    y=0
    y1=0
    #check in all 'KNN' neighbors and their distances aggregated class wise and mean distance found.
    for z in range(KNN):
        if data3[z,2]==1:
            x1=x1+1
            x=x+data3[z,3]
        else:
            y=y+data3[z,3]
            y1=y1+1
        if x1==0:
            return 0
        elif y1==0:
            return 1
        else:
            if (x/x1<=y/y1):
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
plt.title('Decision boundary for KNN 2 neighbors')

