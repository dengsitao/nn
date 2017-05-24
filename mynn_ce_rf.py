from __future__ import division
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

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

def normalize(X):
    return (X-np.mean(X))/np.std(X)

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


alpha = 0.003
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 300
#hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
weight1 = np.random.uniform(-0.1,0.1,(input_dim+1,hidden_dim1))
weight2 = np.random.uniform(-0.1,0.1,(hidden_dim1+1,output_dim))
#imgw=weight1.reshape(1, (input_dim+1)*hidden_dim1)
#showImg(imgw)
#for i in range(785):
#    weight1[i,:]=-0.1+i*0.0003
#for i in range(301):
#    weight2[i,:]=-0.1+i*0.004

#print 'weight1=',weight1
#print 'weight2=',weight2
#weight2 = (2*np.random.random((hidden_dim2+1,output_dim)) -1)
bias_1=np.zeros((1, hidden_dim1))
bias_2=np.zeros((1, output_dim))
#bias_2=np.zeros((1, output_dim))
d1=np.zeros(weight1.shape)
d2=np.zeros(weight2.shape)
#d2=np.zeros(weight2.shape)

imagef = open('./data/train-images-idx3-ubyte', 'rb')
labelf = open('./data/train-labels-idx1-ubyte', 'rb')
magic, imgNum=struct.unpack(">II", imagef.read(8))
imgRow, imgCol =struct.unpack(">II", imagef.read(8))
print magic, imgNum, imgRow, imgCol
lblMagic, lblNum=struct.unpack(">II", labelf.read(8))
print lblMagic, lblNum
overallError = 0
valiNum=10000
trainNum=imgNum-valiNum
Xa=np.zeros((trainNum, imgRow*imgCol))
Ya=np.zeros((trainNum, 1))
Xv=np.zeros((valiNum, imgRow*imgCol))
Yv=np.zeros((valiNum, 1))
print '----start read data----'
for i in range(trainNum):
    Xa[i, range(imgRow*imgCol)]=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    Ya[i, 0]=np.fromfile(labelf, np.uint8, 1)
for i in range(valiNum):
    Xv[i, range(imgRow*imgCol)]=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    Yv[i, 0]=np.fromfile(labelf, np.uint8, 1)
#Xa=sigmoid(Xa)
#Xa=Xa/255
#Xa=(Xa-np.mean(Xa))/np.std(Xa)
Xa=normalize(Xa)
Xv=normalize(Xv)

for i in range(0):
    img=Xa[i].reshape(imgRow, imgCol)
    showImg(img)
    print 'y=',Ya[i]
#img=Xa[1].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[1]
#img=Xa[2].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[2]
imagef.close()
labelf.close()

print '----finish read data----'

loop=imgNum-valiNum
errordot=np.zeros((loop,1))

for k in range(1):
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
        errordot[j]=-math.log(p3[0, y[0]])
        overallError+=errordot[j]
        
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
#plt.plot(range(loop), errordot, "o")
#plt.show()
  

print 'train finish'

imagef.close()
labelf.close()
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
Xt=normalize(Xt)
test_accuracy=predict(Xt, Yt, weight1, weight2)

    

