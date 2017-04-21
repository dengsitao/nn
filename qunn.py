import struct
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(0)
## compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
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

def costFunc(h, y):
    m=h.shape[0]
    return np.sum(pow((h-y),2))/m

def costFuncReg(self, h, y):
    cost=self.costFunc(h, y)
    return cost #+ ((self.lda/2)*(np.sum(pow(self.ih,2)) + np.sum(pow(self.ho,2))))

alpha = 0.03
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 500
hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
synapse_0 = (2*np.random.random((input_dim,hidden_dim1))-1)
synapse_1 = (2*np.random.random((hidden_dim1,hidden_dim2))-1)
synapse_2 = (2*np.random.random((hidden_dim2,output_dim)) -1)
bias_0=np.zeros((1, hidden_dim1))
bias_1=np.zeros((1, hidden_dim2))
bias_2=np.zeros((1, output_dim))
imagef = open('/home/rdeng/code/mine/nn/data/train-images-idx3-ubyte', 'rb')
labelf = open('/home/rdeng/code/mine/nn/data/train-labels-idx1-ubyte', 'rb')

imagef = open('/home/rdeng/code/mine/nn/data/train-images-idx3-ubyte', 'rb')
labelf = open('/home/rdeng/code/mine/nn/data/train-labels-idx1-ubyte', 'rb')
magic, imgNum=struct.unpack(">II", imagef.read(8))
imgRow, imgCol =struct.unpack(">II", imagef.read(8))
print magic, imgNum, imgRow, imgCol
lblMagic, lblNum=struct.unpack(">II", labelf.read(8))
print lblMagic, lblNum
overallError = 0
loop=imgNum
errordot=np.zeros((loop,1))
#for j in range(imgNum):
for j in range(loop):
    #read a 28x28 image and a byte label
    X=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    y=np.fromfile(labelf, np.uint8, 1)
    X=X.reshape(1, imgRow*imgCol)
    #Forward propagation
    z1=np.dot(X, synapse_0)
    f1=sigmoid(z1+bias_0)
    z2=np.dot(f1, synapse_1)
    f2=sigmoid(z2+bias_1)

    z3=np.dot(f2, synapse_2)
    f3=sigmoid(z3+bias_2)
    sumf3=np.sum(f3)
    p3=softmax(f3)
    yy=np.zeros((1, output_dim))
    yy[0,y]=sumf3

    #backward propagation
    error=pow(f3-yy,2)/2
    #error=costFunc(p3, yy)
    #print 'eroor=',error
    overallError+=np.sum(np.abs(error))
    errordot[j]=np.sum(error)

    gprime=sigm_deri(f3)
    delta3=error#*gprime
    delta2=np.dot(delta3, synapse_2.T)*sigm_deri(f2)
    delta1=np.dot(delta2, synapse_1.T)*sigm_deri(f1)
    d2=np.dot(delta3.T, f2)
    dbias_2=delta3
    d1=np.dot(delta2.T, f1)
    dbias_1=delta2
    d0=np.dot(delta1.T, X)
    dbias_0=delta1
    synapse_0-=alpha*(d0.T+lamda*synapse_0)/hidden_dim1
    synapse_1-=alpha*(d1.T+lamda*synapse_1)/hidden_dim2
    synapse_2-=alpha*(d2.T+lamda*synapse_2)/output_dim
    bias_0-=alpha*dbias_0/hidden_dim1
    bias_1-=alpha*dbias_1/hidden_dim2
    bias_2-=alpha*dbias_2/output_dim
print 'overallerror=',overallError
plt.plot(range(loop), errordot, "o")
plt.show()
  

print 'train finish'

imagef.close()
labelf.close()
#recordf_0 = open('/home/rdeng/code/mine/nn/myparam0', 'wb')
#recordf_1 = open('/home/rdeng/code/mine/nn/myparam1', 'wb')
#recordf_2 = open('/home/rdeng/code/mine/nn/myparam2', 'wb')
#synapse_0.tofile(recordf_0)
#synapse_1.tofile(recordf_1)
#synapse_2.tofile(recordf_2)

testImagef = open('/home/rdeng/code/mine/nn/data/t10k-images-idx3-ubyte', 'rb')
testLabelf = open('/home/rdeng/code/mine/nn/data/t10k-labels-idx1-ubyte', 'rb')

tmagic, timgNum=struct.unpack(">II", testImagef.read(8))
timgRow, timgCol =struct.unpack(">II", testImagef.read(8))
print tmagic, timgNum, timgRow, timgCol
tlblMagic, tlblNum=struct.unpack(">II", testLabelf.read(8))
print tlblMagic, tlblNum
#for j in range(imgNum):
rightSum=0
wrongSum=0
for j in range(timgNum):
#for j in range(1):
    #read a 28x28 image and a byte label
    X=np.fromfile(testImagef, np.uint8, timgRow*timgCol)
    y=np.fromfile(testLabelf, np.uint8, 1)
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(y)
    #img = X.reshape(imgRow, imgCol)
    #show(img)
    X=X.reshape(1, imgRow*imgCol)
    #print 'X= ',X.shape
    #print 'y=',y
    #print 'y.shape= ',y.shape, 'y = ', y
    #print 'synapse_0= ',synapse_0.shape
    #Forward propagation
    z1=np.dot(X, synapse_0)
    f1=sigmoid(z1+bias_0)
    #print 'f1= ',f1.shape
    #print 'synapse_1= ',synapse_1.shape
    z2=np.dot(f1, synapse_1)
    f2=sigmoid(z2+bias_1)

    z3=np.dot(f2, synapse_2)
    f3=sigmoid(z3+bias_2)
    p3=softmax(f3)
    dd=np.zeros((1, output_dim))
    indexd=np.argmax(p3)
    #dd[1,indexd]=1
    yy=np.zeros((1, output_dim))
    #yy[0,y]=1
    if indexd==y:
        rightSum+=1
    else:
        wrongSum+=1

print 'right: ',rightSum,'Wrong: ',wrongSum
    

