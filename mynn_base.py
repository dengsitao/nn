import numpy as np

def normalize(X):
    return (X-np.mean(X))/np.std(X)

class layer:
	'one layer with neureons(X), weights and activate functions'
	input_dim = 0
	output_dim = 0
	alpha = 0.01

	def __init__(self, input_dim, output_dim, act_func, act_func_deri, alpha):
		self.alpha=alpha
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.activate_function=act_func
		self.d_act_func=act_func_deri
		self.lyr_weight=np.random.uniform(-0.1,0.1,(input_dim+1,output_dim))

	def forward_prop(self, X):
		self.X=X
		X1=np.c_[1, X]
		a=np.dot(X1, self.lyr_weight)
		self.a1=self.activate_function(a)
		return self.a1

	def backward_prop(self, delta):
		w2=self.lyr_weight[1:,:]
		w2=w2.reshape(self.input_dim, self.output_dim)
		#print  'input_dim=',self.input_dim,'output_dim=',self.output_dim,'delta.shape=',delta.shape,'w2.T.shape=',w2.T.shape,'X.shape=',self.X.shape
		self.delta2=np.dot(delta, w2.T)*self.d_act_func(self.X)
		a21=np.c_[[1], self.X]
		d2=np.dot(delta.T, a21).T
		#print  'input_dim=',self.input_dim,'output_dim=',self.output_dim,'lyr_weight.shape=',self.lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.alpha
		self.lyr_weight-=self.alpha*(d2)
		return self.delta2

class layer_param:
	def __init__(self, input_dim, output_dim, act_func, act_deri, alpha):
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.act_func=act_func
		self.act_deri=act_deri
		self.alpha=alpha

class nnetwork:
	def __init__(self, X, Y, sample_num, layer_num, layer_param, input_dim, output_dim):
        self.train_num=sample_num*9/10
        self.validate_num=sample_num/10
        self.trainX=X[0:self.train_num,:]
        self.trainY=Y[0:self.train_num,:]
        self.validateX=Xa[self.train_num:sample_num,:]
        self.validateY=Ya[self.train_num:sample_num,:]
		self.layer_num=layer_num
		self.layers=[]
		for i in range(layer_num):
			self.layers.append(layer(layer_param[i].input_dim, layer_param[i].output_dim, layer_param[i].act_func, layer_param[i].act_deri, layer_param[i].alpha))
			self.layers[i].weights=np.random.uniform(-0.1,0.1,(layer_param[i].input_dim+1,layer_param[i].output_dim))
        self.sample_size=input_dim
        self.output_dim=output_dim
        
    def train():
        for i in range(self.sample_num):
            X=self.trainX[i]
            X=X.reshape(1, self.sample_size)
            y=self.trainY[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, y[0]]=1.0
            for j in range(self.layer_num):
                X=self.layers[j].forward_prop(X)
            error=X-yy
            delta=error
            for j in xrange(self.layer_num):
                delta=self.layers[j].backward_prop(delta)
        print '----training finish----'
        validate()

    def validate():
        right=0
        wrong=0
        for i in range(self.validate_num):
            X=self.validateX[i]
            X=X.reshape(1, self.sample_size)
            y=self.validateY[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, y[0]]=1.0
            for j in range(self.layer_num):
                X=self.layers[j].forward_prop(X)
            index=np.argmax(X)
            if index==y:
                right+=1
            else:
                wrong+=1
        print '----training finish----'















