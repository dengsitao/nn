from __future__ import division
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

import mynn_base as mybs
import mynn_utils as myutils
np.random.seed(0)
## compute sigmoid nonlinearity
def sigmoid(x):
    output = 1.0/(1.0+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigm_deri(output):
    return output*(1-output)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    #exp_scores=np.exp(x)
    #probs=exp_scores/np.sum(exp_scores, axis=1,keepdims=True)
    #return probs

def softmax_deri(signal):
    J = - signal[..., None] * signal[:, None, :] # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = signal * (1. - signal) # diagonal
    return J.sum(axis=1) # sum across-rows for each sample

def relu(x):
    return np.maximum(x, 0)

def relu_deri(output):
    return 1.*(output>0)


def showImg(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    #from matplotlib import pyplot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

def forwardProp(X, weight, activate_func):
    X1=np.c_[1, X]
    Z1=np.dot(X1, weight)
    Res1 = activate_func(Z1)
    return Res1

# def normalize(X):
#     return (X-np.mean(X))/np.std(X)

def predict(Xi, Yi, weight1, weight2):
    num=Yi.size
    rightSum=0
    wrongSum=0
    for j in range(num):
        #for j in range(1):
        #read a 28x28 image and a byte label
        #X=Xa[j+imgNum-valiNum]
        X=Xi[j]
        X=X.reshape(1,28*28)
        y=Yi[j]
        #y=Ya[j+imgNum-valiNum]
        #Forward propagation
        a2=forwardProp(X, weight1, sigmoid)
        a3=forwardProp(a2, weight2, softmax)
        indexd=np.argmax(a3)
        if indexd==y:
            rightSum+=1
        else:
            wrongSum+=1

    accuracy=rightSum/num
    print 'train',k,' right: ',rightSum,'Wrong: ',wrongSum, accuracy, '%'
    return accuracy

# def loadImgData(imgfile, imgNum, count, offset):
#     if count > imgNum:
#         print 'count=',count,' > ','imgNum=', imgNum
#         count=imgNum
#     Xa=np.zeros((count, imgRow*imgCol))
#     imgfile.seek(offset*imgRow*imgCol+16)
#     for i in range(count):
#         Xa[i, range(imgRow*imgCol)]=np.fromfile(imgfile, np.uint8, imgRow*imgCol)
#     return Xa

# def loadLabelData(lblfile, imgNum, count, offset):
    
#     if count > lblNum:
#         print 'count=',count,' > ','lblNum=', lblNum
#         count=imgNum
#     Ya=np.zeros((count, 1))
#     lblfile.seek(offset+8)
#     for i in range(count):
#         Ya[i, 0]=np.fromfile(lblfile, np.uint8, 1)
#     return Ya

alpha = 0.003
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 300
#hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
weight1 = np.random.uniform(-0.1,0.1,(input_dim+1,hidden_dim1))
weight2 = np.random.uniform(-0.1,0.1,(hidden_dim1+1,output_dim))

bias_1=np.zeros((1, hidden_dim1))
bias_2=np.zeros((1, output_dim))

d1=np.zeros(weight1.shape)
d2=np.zeros(weight2.shape)


imgNum, imgRow, imgCol, lblNum, Xa, Ya = myutils.loadMNISTData()
valiNum=int(imgNum/10)
Xv=Xa[imgNum-valiNum-1:imgNum,:]
Yv=Ya[lblNum-valiNum-1:lblNum,:]


print '----finish read data----'
epoch=3
loop=imgNum-valiNum
errordot=np.zeros((loop*epoch,1))

for k in range(epoch):
    overallError=0
    for j in range(loop):
        X=Xa[j]
        X=X.reshape(1,imgRow*imgCol)
        y=Ya[j]
        #Forward propagation
        a2=forwardProp(X, weight1, sigmoid)

        p3=forwardProp(a2, weight2, softmax)

        yy=np.zeros((1, output_dim))
        yy[0.0,y[0]]=1.0
    
        #backward propagation
        error=p3-yy
        errordot[k*loop+j]=-math.log(p3[0, y[0]])
        overallError+=errordot[k*loop+j]
        
        #print 'delta3=',delta3.shape,'weight2.T=',weight2.T.shape,'f2=',f2.shape
        delta3=error#*gprime
        w2=weight2[1:,:]
        w2=w2.reshape(hidden_dim1, output_dim)
        #print  'delta3=',delta3.T,'w2.T=',w2.T.shape,'a2=',a2.shape
        delta2=np.dot(delta3, w2.T)*sigm_deri(a2)
        #print 'delta2=',delta2.shape,'weight2.T=',weight2.T.shape,'a3=',a3.shape
        #delta1=np.dot(delta2, weight1.T)*sigm_deri(a2)
        #print 'd2=',d2.shape,'delta3.T=',delta3.T.shape,'f2=',f2.shape
        #print 'weight2=',weight2.shape,'d2=',d2.shape,'delta3.T=',delta3.T.shape,'a2=', a2.shape
        a21=np.c_[[1], a2]
        d2=np.dot(delta3.T, a21).T
        #dbias_1=delta2
        #print 'weight1=',weight1.shape,'d1=',d1.shape,'delta2.T=',delta2.T.shape,'a1=', a1.shape
        #delta2=delta2[:,1:]
        #delta2=delta2.reshape(1, hidden_dim1)
        a1=np.c_[[1],X]
        d1=np.dot(delta2.T, a1).T
        #weight1-=alpha*(d1)#+lamda*weight1)
        #weight2-=alpha*(d2)#+lamda*weight2)
        weight1-=alpha*(d1)#+lamda*weight1)
        weight2-=alpha*(d2)#+lamda*weight2)
    print '----train: ',k,' finish----'
    print 'overallerror=',overallError
    accuracy=predict(Xv, Yv, weight1, weight2)

    if (accuracy > 0.95):
        break
    #plt.plot(range(loop), errordot, "o")
    #plt.show()
#print 'overallerror=',overallError
#plt.plot(range(epoch*loop), errordot)
#plt.show()
  

print 'train finish'

#imagef.close()
#labelf.close()
#recordf_0 = open('/home/rdeng/code/mine/nn/myparam0', 'wb')
#recordf_1 = open('/home/rdeng/code/mine/nn/myparam1', 'wb')
#recordf_2 = open('/home/rdeng/code/mine/nn/myparam2', 'wb')
#weight0.tofile(recordf_0)
#weight1.tofile(recordf_1)
#weight2.tofile(recordf_2)

timagef = open('./data/t10k-images-idx3-ubyte', 'rb')
tlabelf = open('./data/t10k-labels-idx1-ubyte', 'rb')

tmagic, timgNum=struct.unpack(">II", timagef.read(8))
timgRow, timgCol =struct.unpack(">II", timagef.read(8))
print tmagic, timgNum, timgRow, timgCol
tlblMagic, tlblNum=struct.unpack(">II", tlabelf.read(8))
print tlblMagic, tlblNum

Xt=np.zeros((timgNum, timgRow*imgCol))
Yt=np.zeros((tlblNum, 1))
print '----start read data----'
for i in range(timgNum):
    Xt[i, range(timgRow*timgCol)]=np.fromfile(timagef, np.uint8, timgRow*timgCol)
    Yt[i, 0]=np.fromfile(tlabelf, np.uint8, 1)
#Xt=sigmoid(Xt)
#Xt=(Xt-np.mean(Xt))/np.std(Xt)
Xt=mybs.normalize(Xt)
test_accuracy=predict(Xt, Yt, weight1, weight2)

    

