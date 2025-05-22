# This code is used to transfer the data from the source file to the destination file.

# 实现对已经训练过的模型进行使用自己的数据进行迁移学习的过程。


# 导入必要的库

import os
import numpy as np
from dataset.dataloader import DataLoaderCV
from model.mlp import Dense, MLP
from loss.mseloss import MSELoss


layers = [
    Dense(input_size=784, output_size=256, activation='tanh'),
    Dense(input_size=256, output_size=128, activation='relu'),
    Dense(input_size=128, output_size=256, activation='relu'),
    Dense(input_size=256, output_size=10, activation='sigmoid')
]
# 创建mlp
mlp = MLP(layers)
# 测试模型
mlp.load_model('result/tl/mpl_2000.npz') # 加载模型权重

# 加载数据集
root_dir = 'data/'
train_list = np.load('data/train_list.npz')['train_list']
test_list = np.load('data/test_list.npz')['test_list']

# print('train_list:', train_list)
# print('test_list:', test_list)
# print('train_list:', train_list.files)
# print('test_list:', test_list.files)
# print('train_list[\'data\'].shape:', train_list['train_list'].shape)
# print('test_list[\'data\'].shape:', test_list['test_list'].shape)

train_loader = DataLoaderCV(root_dir, train_list, batch_size=20, shuffle=True)
test_loader = DataLoaderCV(root_dir, test_list, batch_size=10, shuffle=False)

# print(len(train_loader), len(test_loader))
# for i, (x, y) in enumerate(train_loader):
#     print(x.shape, y.shape)

print('开始迁移学习')

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
            model.backward(criterion.backward() * 100, learning_rate)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss / len(train_loader)}')

# 预测模型
def predict(model, x_test):
    return model.forward(x_test)


if __name__ == '__main__':

    criterion = MSELoss()
    epcohs = 2000
    learning_rate = 0.001
    
    # train(mlp, criterion, epcohs, learning_rate)
    
    # # 保存模型的权重
    # save_path = 'result/tl'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)    
    # mlp.save_model(os.path.join(save_path, f'mpl_{epcohs}.npz'))
    
    # # 将mlp删除
    del mlp
    # 创建新的mlp
    mlp = MLP(layers)
    # 测试模型
    mlp.load_model(f'result/tl/mpl_{epcohs}.npz') # 加载模型权重
    accuracy = 0
    for batch_images, batch_labels in test_loader:
        output = predict(mlp, batch_images)
        # print(output[0], batch_labels[0])
        # print(output.argmax(axis=1), batch_labels.argmax(axis=1))
        accuracy += (output.argmax(axis=1) == batch_labels.argmax(axis=1)).sum() / len(batch_labels)
    print(f'Accuracy: {accuracy/len(test_loader)}')