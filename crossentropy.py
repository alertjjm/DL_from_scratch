import sys, os
import numpy as np
from math import exp
from PIL import Image
import pickle
from DL_from_scratch.mnist import load_mnist
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network
def cross_entropy_error(y,t):
    print(y.ndim)
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
def softmax(x):
    c=np.max(x)
    exp_a=np.exp(x-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
def sigmoid(x):
    return 1/(1+np.exp(-x))
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)

x,t=get_data()
network=init_network()
accuracy_cnt=0
batch_size=1
batchmask=np.random.choice(x.shape[0],batch_size)
x_batch=x[batchmask]
t_batch=t[batchmask]
y_batch=predict(network,x_batch)
print(cross_entropy_error(np.array([0.1]),np.array([0.2])))
