import numpy as np

class ThreeLayerNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim=1, learning_rate=0.01):
        """
        初始化三层神经网络的参数
        网络结构: Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid)
        """
        self.lr = learning_rate
        
        # 为了防止梯度消失或爆炸，使用 He 初始化 (针对 ReLU) 或简单的标准正态分布缩小
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(hidden1_dim, input_dim) * np.sqrt(2. / input_dim),
            'b1': np.zeros((hidden1_dim, 1)),
            'W2': np.random.randn(hidden2_dim, hidden1_dim) * np.sqrt(2. / hidden1_dim),
            'b2': np.zeros((hidden2_dim, 1)),
            'W3': np.random.randn(output_dim, hidden2_dim) * 0.01,
            'b3': np.zeros((output_dim, 1))
        }
        
        # 用于存储前向传播的中间变量，反向传播时需要用到
        self.cache = {}

    # --- 激活函数 ---
    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_backward(self, dA, Z):
        """ReLU 的导数：当 Z > 0 时为 1，否则为 0"""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid(self, Z):
        # 裁剪 Z 以防止 exp 溢出
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    # --- 前向传播 ---
    def forward(self, X):
        """
        X 形状: [特征维度 input_dim, 样本数量 m]
        """
        # 第一层 (Hidden 1)
        Z1 = np.dot(self.params['W1'], X) + self.params['b1']
        A1 = self.relu(Z1)
        
        # 第二层 (Hidden 2)
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        A2 = self.relu(Z2)
        
        # 第三层 (Output)
        Z3 = np.dot(self.params['W3'], A2) + self.params['b3']
        A3 = self.sigmoid(Z3)
        
        # 缓存中间变量
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
        return A3

    # --- 损失函数 ---
    def compute_loss(self, A3, Y):
        """
        计算二元交叉熵损失
        Y 形状: [1, 样本数量 m]
        """
        m = Y.shape[1]
        # 添加极小值 epsilon 防止 log(0)
        epsilon = 1e-15
        A3 = np.clip(A3, epsilon, 1 - epsilon)
        
        loss = - (1. / m) * np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
        return np.squeeze(loss)

    # --- 反向传播 ---
    def backward(self, Y):
        """
        根据计算图逆向计算梯度
        """
        m = Y.shape[1]
        
        # 提取缓存的数据
        X = self.cache['X']
        A1, Z1 = self.cache['A1'], self.cache['Z1']
        A2, Z2 = self.cache['A2'], self.cache['Z2']
        A3 = self.cache['A3']
        W2, W3 = self.params['W2'], self.params['W3']
        
        gradients = {}
        
        # 1. 输出层的梯度 (Sigmoid + BCE Loss 的绝妙化简)
        # dL/dZ3 = A3 - Y
        dZ3 = A3 - Y
        gradients['dW3'] = (1. / m) * np.dot(dZ3, A2.T)
        gradients['db3'] = (1. / m) * np.sum(dZ3, axis=1, keepdims=True)
        
        # 2. 第二个隐藏层的梯度
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = self.relu_backward(dA2, Z2)
        gradients['dW2'] = (1. / m) * np.dot(dZ2, A1.T)
        gradients['db2'] = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # 3. 第一个隐藏层的梯度
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = self.relu_backward(dA1, Z1)
        gradients['dW1'] = (1. / m) * np.dot(dZ1, X.T)
        gradients['db1'] = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return gradients

    # --- 参数更新 (梯度下降) ---
    def update_parameters(self, gradients):
        self.params['W1'] -= self.lr * gradients['dW1']
        self.params['b1'] -= self.lr * gradients['db1']
        self.params['W2'] -= self.lr * gradients['dW2']
        self.params['b2'] -= self.lr * gradients['db2']
        self.params['W3'] -= self.lr * gradients['dW3']
        self.params['b3'] -= self.lr * gradients['db3']

# --- 测试与训练循环 ---
if __name__ == "__main__":
    # 构造简单的异或 (XOR) 或非线性可分数据集
    # 样本数 m=4, 特征维度 nx=2
    X_train = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]]) # 形状: [2, 4]
    Y_train = np.array([[0, 1, 1, 0]]) # 形状: [1, 4]

    # 实例化网络 (2 -> 4 -> 4 -> 1)
    nn = ThreeLayerNetwork(input_dim=2, hidden1_dim=4, hidden2_dim=4, output_dim=1, learning_rate=0.1)

    epochs = 5000
    for i in range(epochs):
        # 1. 前向传播
        predictions = nn.forward(X_train)
        
        # 2. 计算 Loss
        loss = nn.compute_loss(predictions, Y_train)
        
        # 3. 反向传播
        grads = nn.backward(Y_train)
        
        # 4. 梯度下降更新
        nn.update_parameters(grads)
        
        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    print("\n训练完成。最终预测结果 (应接近 0, 1, 1, 0):")
    print(nn.forward(X_train))