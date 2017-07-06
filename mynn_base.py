import numpy as np

def normalize(X):
    return (X-np.mean(X))/np.std(X)

class layer:
	'one layer with neureons(X), weights and activate functions'
	lyr_width = 0
	lyr_height = 0
	lyr_depth = 1
	

	def __init__(self, width, height, depth, act_func):
		lyr_width=width
		lyr_height=height
		lyr_depth=depth
		activate_function=act_func
		lyr_X=np.zeros(1, width*height)

		lyr_weight=np.zeros(1, width*height+1)

	def forward_prop():
		X=np.c_[1, self.lyr_X]
		a=np.dot(X, self.lyr_weight)
		a1=self.activate_function(a)
		return a1
