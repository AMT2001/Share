

import os
from dataset.dataloader import load_images, load_labels, DataLoader
from loss.mseloss import MSELoss
from model.mlp import MLP, Dense


# 加载train_images和train_labels
train_images = load_images('data/train-images.idx3-ubyte', 60000)
train_labels = load_labels('data/train-labels.idx1-ubyte', 60000)
test_images = load_images('data/t10k-images.idx3-ubyte', 10000)
test_labels = load_labels('data/t10k-labels.idx1-ubyte', 10000, one_hot=False)

# 创建DataLoader实例
batch_size = 60
train_loader = DataLoader(train_images, train_labels, batch_size=batch_size, shuffle=True)
batch_size = 10
# 定义训练集和测试集
test_loader = DataLoader(test_images, test_labels, batch_size=batch_size, shuffle=False)


# 定义网络结构
layers = [
    Dense(input_size=784, output_size=256, activation='tanh'),
    Dense(input_size=256, output_size=128, activation='softmax'),
    Dense(input_size=128, output_size=256, activation='softmax'),
    Dense(input_size=256, output_size=10, activation='sigmoid')
]

# 训练模型
def train(model, criterion, epochs, learning_rate):
    for epoch in range(epochs):
        loss = 0
        # 使用DataLoader来迭代获取批次数据
        for batch_images, batch_labels in train_loader:
            output = model.forward(batch_images)
            # 将结果进行概率归一化
            # output = output / output.sum(axis=1, keepdims=True)
            loss += criterion.forward(output, batch_labels) 
            model.backward(criterion.backward(), learning_rate)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss / len(train_loader)}')


# 预测模型
def predict(model, x_test):
    return model.forward(x_test)


if __name__ == '__main__':
    mlp = MLP(layers)
    criterion = MSELoss()
    train(mlp, criterion, 100, 0.001)
    # 保存模型的权重
    save_path = 'result/mpl'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mlp.save_model(save_path)
    
    # 将mlp删除
    del mlp
    
    # 创建新的mlp
    mlp = MLP(layers)
    # 测试模型
    mlp.load_model('result/mpl.npz') # 加载模型权重
    accuracy = 0
    for batch_images, batch_labels in test_loader:
        output = predict(mlp, batch_images)
        accuracy += (output.argmax(axis=1) == batch_labels).sum() / len(batch_labels)
    print(f'Accuracy: {accuracy/len(test_loader)}')
    
        