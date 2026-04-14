import numpy as np
import Layer as nn

# 测试全连接层
Linear = nn.Dense(in_features=4, out_features=2)
print(Linear.params["w"].shape, Linear.params["b"].shape)
inputs = np.array([1.0, 2.0, 3.0, 4.0])
output = Linear.forward(inputs)
print("Linear forward output:", output)

grad_before = np.ones_like(output)
grad = Linear.backward(grad_before)
print("Linear backward output:", grad) # 形状应该是 (4,) 对应输入的维度

relu = nn.Relu()
relu_output = relu.forward(output)
print("ReLU forward output:", relu_output)
relu_grad = relu.backward(grad_before)
print("ReLU backward output:", relu_grad) # 形状应该和 output 一

y = np.array([1.0, 2.0, 3.0, 4.0])
pred = np.array([[1.0, 0.0, 0.0, 0.0],
              [1.0, 2.0, 0.0, 0.0],
              [1.0, 2.0, 3.0, 0.0],
              [1.0, 2.0, 3.0, 4.0]])
from Loss import CrossEntropyLoss as loss

loss = loss(pred, y)
print(loss.loss())
gradients =loss.grad()
print(gradients)