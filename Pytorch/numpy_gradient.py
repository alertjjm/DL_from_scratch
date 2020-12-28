import torch
import torchvision
import numpy as np
N, D_in, H, D_out=64,1000,100,10
x=np.random.randn(N, D_in)
y=np.random.randn(N,D_out)
w1=np.random.randn(D_in, H)
w2=np.random.randn(H,D_out)
learning_rate=1e-6
for i in range(500):
    h=x.dot(w1)
    h_relu=np.maximum(h,0)#relu이니까 0보다 작으면 다 0이고 0보다 크면 큰것, 원래 두 넘파이배열이 인지여야 하는데 여기선 0이 브로드캐스팅
    y_pred=h_relu.dot(w2)
    loss=np.square(y_pred-y).sum()
    print(i, loss)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2