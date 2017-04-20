import struct
import numpy as np

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
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

alpha = 0.1
input_dim = 28*28
hidden_dim1 = 500
hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim1)) - 1
synapse_1 = 2*np.random.random((hidden_dim1,hidden_dim2)) - 1
synapse_2 = 2*np.random.random((hidden_dim2,output_dim)) - 1
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_2_update = np.zeros_like(synapse_2)
bias_0=np.zeros((1, hidden_dim1))
bias_1=np.zeros((1, hidden_dim2))
bias_2=np.zeros((1, output_dim))

imagef = open('/home/rdeng/code/mine/nn/data/train-images-idx3-ubyte', 'rb')
labelf = open('/home/rdeng/code/mine/nn/data/train-labels-idx1-ubyte', 'rb')

magic, imgNum=struct.unpack(">II", imagef.read(8))
imgRow, imgCol =struct.unpack(">II", imagef.read(8))
print magic, imgNum, imgRow, imgCol
lblMagic, lblNum=struct.unpack(">II", labelf.read(8))
print lblMagic, lblNum
overallError = 0
for j in range(imgNum):
#for j in range(10):
    #read a 28x28 image and a byte label
    X=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    y=np.fromfile(labelf, np.uint8, 1)
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
    #f1=sigmoid(z1+bias_0)
    f1=relu(z1+bias_0)
    #print 'f1= ',f1.shape
    #print 'synapse_1= ',synapse_1.shape
    z2=np.dot(f1, synapse_1)
    f2=relu(z2+bias_1)

    z3=np.dot(f2, synapse_2)
    f3=sigmoid(z3+bias_2)
    #p3=-np.log(softmax(f3))
    p3=softmax(f3)
    #p3=softmax(f2)
    yy=np.zeros((1, output_dim))
    yy[0,y]=1
    #print 'yy=', yy
    #print 'f3=',f3
    #corect_logprobs = -np.log(f3[range(output_dim),y])
    #print 'f3=',f3.shape,'corect_logprobs=',corect_logprobs.shape
    #f3=corect_logprobs
    #data_loss = np.sum(corect_logprobs)/1
    #dscores =f3 
    #dscores[0,y] -= 1
    
    #dW = np.dot(X.T, dscores)
    #db = np.sum(dscores, axis=0, keepdims=True)
    #pred = np.argmax(f3)
    #print 'guess = ',pred, 'label = ',y
    #print 'z2=',z2.shape
    #print 'f2=',f2.shape
    #backward propagation
    error=p3-yy
    #print 'y=',y, 'f3=',f3, 'error=',error
    gprime=sigm_deri(f3)
    #gprime=softmax_deri(f3)
    #print 'gprime=',gprime.shape,'=sigma_deri(f3), f3=',f3.shape
    #error=data_loss
    delta3=error*gprime
    #print 'delta3=',delta3.shape,'sy_2.T=',synapse_2.T.shape
    delta2=np.dot(delta3, synapse_2.T)*relu_deri(f2)
    #print 'delta2=',delta2.shape,'delta3xs2.T,s2.T=',synapse_2.T.shape,'sgder(f2)=',sigm_deri(f2).shape
    delta1=np.dot(delta2, synapse_1.T)*relu_deri(f1)
    #print 'delta1=',delta1.shape,'delta2xs1,s1=',synapse_1.T.shape,'sgder(f1)=',sigm_deri(f1).shape

    #print 'delta3=',delta3.shape,'f2.T=',f2.T.shape
    d2=np.dot(f2.T,delta3)
    dbias_2=delta3
    #print 'delta2=',delta2.shape,'f2.T=',f2.T.shape,'d2=',d2.shape
    #print 'delta2=',delta2.shape,'f1.T=',f1.T.shape
    d1=np.dot(f1.T, delta2)
    dbias_1=delta2
    #print 'delta1=',delta1.shape,'f1.T=',f1.T.shape,'d1=',d1.shape
    #print 'delta1=',delta1.shape,'X.T=',X.T.shape
    d0=np.dot(X.T, delta1)
    dbias_0=delta1
    #print 'delta1=',delta1.shape,'X=',X.T.shape,'d0=',d0.shape
    synapse_0-=alpha*d0
    synapse_1-=alpha*d1
    synapse_2-=alpha*d2
    bias_0-=alpha*dbias_0
    bias_1=alpha*dbias_1
    bias_2=alpha*dbias_2
    #print 'guess = ',pred, 'label = ',y,' error=',error

    #if(j%1000 == 0):
    #    print 'error=',error

print 'train finish'

testImagef = open('/home/rdeng/code/mine/nn/data/train-images-idx3-ubyte', 'rb')
testLabelf = open('/home/rdeng/code/mine/nn/data/train-labels-idx1-ubyte', 'rb')

tmagic, timgNum=struct.unpack(">II", testImagef.read(8))
timgRow, timgCol =struct.unpack(">II", testImagef.read(8))
print tmagic, timgNum, timgRow, timgCol
tlblMagic, tlblNum=struct.unpack(">II", testLabelf.read(8))
print tlblMagic, tlblNum
#for j in range(imgNum):
rightSum=0
wrongSum=0
for j in range(timgNum):
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
    f1=relu(z1+bias_0)
    #print 'f1= ',f1.shape
    #print 'synapse_1= ',synapse_1.shape
    z2=np.dot(f1, synapse_1)
    f2=relu(z2+bias_1)

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
    

