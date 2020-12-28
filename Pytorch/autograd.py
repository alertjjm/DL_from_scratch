import torch
dtype=torch.float
device=torch.device("cpu")
# device = torch.device("cuda:0")
N, D_in, H, D_out = 64, 1000, 100, 10

x=torch.randn(N,D_in,device=device, dtype=dtype)
y=torch.randn(N,D_out, device=device, dtype=dtype)
w1=torch.randn(D_in,H,device=device, dtype=dtype, requires_grad=True) #해당 Tensor에 대한 변화도 계산해주세요
w2=torch.randn(H,D_out,device=device, dtype=dtype,requires_grad=True)  #해당 Tensor에 대한 변화도 계산해주세요

learning_rate=1e-6
for i in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2) #grad를 자동으로 계산해주기때문에 중간값을 갖고있을 필요 없이 한번에 수식으로 정리
    loss = (y_pred - y).pow(2).sum()
    if i % 100 == 99:
        print(loss.item())
    loss.backward() #loss tensor에서 시작해서 역전파 보내기
    with torch.no_grad(): #이제 auto grad 안한단 소리
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
