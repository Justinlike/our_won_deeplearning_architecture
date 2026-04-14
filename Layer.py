import numpy as np

# layer
# 上面流程代码中 model 进行 forward 和 backward，其实底层都是网络层在进行实际运算，
# 因此网络层需要有提供 forward 和 backward 接口进行对应的运算。
# 同时还应该将该层的参数和梯度记录下来。先实现一个基类如下

class Layer():
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.grads = {}

    # forward 方法接收上层的输入 inputs，实现 的运算
    def forward(self, *inputs):
        raise NotImplementedError

    # backward 的方法接收来自上层的梯度，计算关于参数 和输入的梯度，然后返回关于输入的梯度。
    def backward(self, *grads):
        raise NotImplementedError

class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError

# 之后我们就可以在这个基类的基础上实现各种各样的网络层了，比如全连接层，卷积层，池化层等等。
class Dense(Layer):
    def __init__(self, in_features, out_features,
                 w_init = np.random.random, b_init = np.zeros):
        super(Dense, self).__init__("Linear")
        self.params = {
            "w": w_init([in_features, out_features]),
            "b": b_init([1, out_features])
        }

    def forward(self, inputs:np.ndarray):
        # 将 inputs 保证为二维 (batch, in_features)
        self.inputs = np.atleast_2d(inputs)

        return self.inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad:np.ndarray):
        # 保证 grad 为二维：(batch, out_features)
        grad_arr = np.atleast_2d(grad)

        self.grads["w"] = self.inputs.T @ grad_arr
        self.grads["b"] = np.sum(grad_arr, axis=0, keepdims=True)

        return grad_arr @ self.params["w"].T

class Linear(Layer):
    def __init__(self, name, in_features, out_features):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重和偏置
        self.params = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features)
        self.grads = np.zeros_like(self.params)

    def forward(self, inputs):
        self.inputs = inputs
        return self.params @ inputs + self.bias

    def backward(self, grad_output):
        # 计算权重梯度
        self.grads = self.params @ grad_output + self.bias
        # 计算输入梯度
        grad_input = self.params.T @ grad_output
        return grad_input

class Conv2d(Layer):
    def __init__(self, name, in_channesl, out_channesl, kernel_size, stride=1, padding=0):
        super().__init__(name)
        self.in_channesl = in_channesl
        self.out_channesl = out_channesl
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 初始化卷积核权重和偏置
        self.params = np.random.randn(out_channesl, in_channesl, kernel_size, kernel_size)
        self.grads = np.zeros_like(self.params)
        self.bias = np.random.randn(out_channesl)

    def forward(self, inputs):
        self.inputs = np.atleast_3d(inputs) # 确保输入为三维 (batch, in_channels, height, width)
        # 卷积操作的前向传播实现

        pass

    def backward(self, grad_output):
        # 卷积操作的反向传播实现
        pass

class Pool2d(Layer):
    def __init__(self, name, kernel_size, stride=1, padding=0):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, inputs):
        # 池化操作的前向传播实现
        pass

    def backward(self, grad_output):
        # 池化操作的反向传播实现
        pass

class Sigmoid(Layer):
    def __init__(self, name, x):
        super().__init__(name)

    def forward(self, x):
        return 1 / (1 + np.exp(-self.x))

    def backward(self, grad_output):
        return  self.forward() * (1 - self.forward()) * grad_output



class ReLU(Activation):
    def __init__(self):
        super().__init__("Relu")

    def func(self, x):
        return np.maximum(0, x)

    def derivative_func(self, x):
        return x > 0.0


