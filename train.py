import numpy as np


r=2
y,x,z = np.ogrid[-r:r+1, -r:r+1, -r:r+1,]
mask = x**2 + y**2 + z**2 <= r**2

print(mask)