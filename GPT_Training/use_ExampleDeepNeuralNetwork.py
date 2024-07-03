from ExampleDeepNeuralNetwork import ExampleDeepNeuralNetwork
import torch

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # 为初始权重指定随机种子以确保可重现性
model_without_shortcut = ExampleDeepNeuralNetwork(
     layer_sizes, use_shortcut=False
)

import torch.nn as nn
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # 根据目标和输出的接近程度计算损失
    # 输出是怎样的形式
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播以计算梯度
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 输出权重的平均绝对梯度
            print(f"{name} has gradient mean of {param.grad.abs().mean()}")



print_gradients(model_without_shortcut, sample_input)
print("分割线")
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)