import numpy as np

class Net(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        all_grads = []
        for layer in self.layers:
            grad = layer.backward(grad)
            all_grads.append(layer.grads)

    def get_params_and_grads(self):
        for layer in self.layers:
            yield layer.params, layer.grads

    def get_params(self):
        return [layer.params for layer in self.layers]

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            for key in layer.params.keys():
                layer.params[key] = params[i][key]

class Model(object):
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, targets):
        Loss = self.loss(preds, targets)
        loss = Loss.loss()
        grad = Loss.grad()
        self.net.backward(grad)
        return loss