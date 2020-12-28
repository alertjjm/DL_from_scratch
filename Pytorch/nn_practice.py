import torch
N, D_in, H, D_out = 64, 1000, 100, 10
dtype=torch.float
device=torch.device("cpu")
# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x=torch.randn(N,D_in,device=device, dtype=dtype)
y=torch.randn(N,D_out, device=device, dtype=dtype)
model=torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)
loss_fn=torch.nn.MSELoss(reduction='sum')   #차원을 sum으로 감소시킨다
learning_rate=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()