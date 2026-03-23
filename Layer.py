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

# 之后我们就可以在这个基类的基础上实现各种各样的网络层了，比如全连接层，卷积层，池化层等等。
class Linear(Layer):
    def __init__(self, name, in_features, out_features):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重和偏置
        self.params = np.random.randn(out_features, in_features)
        self.grads = np.zeros_like(self.params)

    def forward(self, inputs):
        self.inputs = inputs
        return self.params @ inputs