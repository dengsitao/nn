#!/usr/bin/env python

import numpy as np

ar2d = [[1,2,3],[4,5,6],[7,8,9]]
ar2d2 = [[1,2,3],[4,5,6],[7,8,9]]
#ar2d2 = np.random.random((2,3))

print ar2d
print ar2d2

#ar2d.append(ar2d2)
ar3d=np.dot(ar2d, ar2d2)
print ar3d

x=-2
y=5
z=-4
step=1

for i in range(1):
    q=x+y
    f=q*z
    dfdq=z
    dfdz=q
    dfdx=1*dfdq
    dfdy=1*dfdq
    if f>0:
        x-=step*dfdx
        y-=step*dfdy
        z-=step*dfdz
    else:
        x+=step*dfdx
        y+=step*dfdy
        z+=step*dfdz
    print('dfdx = ', dfdx)
    print('dfdy = ', dfdy)
    print('dfdz = ', dfdz)
print 'f    = ', f



