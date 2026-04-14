import numpy as np

class BaseLoss(object):
    def __init__(self, pred, y, **kwargs):
        self.pred = np.atleast_2d(pred)
        self.y = self._2onehot(y, self.pred.shape[1])

    def _2onehot(self, y, num_classes):
        # y: (N,) class index -> (N, C)
        if y.ndim == 1:
            one_hot = np.zeros((y.shape[0], num_classes), dtype=np.int8)

            rows = np.arange(y.shape[0])  # (N, ) -> 生成行索引 [0, 1, 2, ..., N-1]
            cols = y.astype(np.int32) -1  # 生成列索引 [y[0], y[1], ..., y[N-1]]

            one_hot[rows, cols] = 1
            return one_hot
        return y.astype(np.int32)

    def loss(self, ):
        raise NotImplementedError

    def grad(self, ):
        raise NotImplementedError

class CrossEntropyLoss(BaseLoss):
    def __init__(self,pred, y, weights=None):
        super(CrossEntropyLoss, self).__init__(pred, y)
        self.weights = weights

    def _softmax(self, logits):
        # 数值稳定版
        logits = logits - np.max(logits, axis=1, keepdims=True) # normalize
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def loss(self):
        # pred, y shape should be (N, C)

        eps = 1e-12
        m = self.pred.shape[0]
        return -np.sum(self.y * np.log(self.pred + eps)) / m

    def grad(self, ):
        # 交叉熵损失函数的梯度计算

        probs = self._softmax(self.pred)
        m = probs.shape[0]
        grad = np.copy(probs)
        return (probs - self.y) / m
