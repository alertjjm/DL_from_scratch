import numpy as np
from matplotlib import pyplot as plt
x=np.arange(-5.0,5.0,0.1)
y=np.array(x>0,dtype=int)
plt.plot(x,y)
plt.show()
