
import os
import cv2
import numpy as np
from model.mlp import Dense, MLP

# # 读取npz文件
# data = np.load('result/mpl.npz')

# # 打印npz文件中的键值对
# print(data.files)

# # 根据键值对获取数据
# for key in data.files:
#     print(key, data[key]) 
#     print(data[key].shape)  # 打印数据形状

# layers = [
#     Dense(input_size=784, output_size=256, activation='tanh'),
#     Dense(input_size=256, output_size=128, activation='relu'),
#     Dense(input_size=128, output_size=256, activation='relu'),
#     Dense(input_size=256, output_size=10, activation='sigmoid')
# ]
# # 创建mlp
# mlp = MLP(layers)
# # 测试模型
# mlp.load_model('result/tl/mpl.npz') # 加载模型权重


# # 读取图片并预测
# img = cv2.imread('data/9/20241206182848.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape(1, 784)
# img = img / 255.0
# output = mlp.forward(img)
# print(output)
# print(output.argmax(axis=1))


# img = cv2.imread('test/9/img_5.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape(1, 784)
# img = img / 255.0
# output = mlp.forward(img)
# print(output)
# print(output.argmax(axis=1))
    
# for index, (batch_images, batch_labels) in enumerate(test_loader):
#     # 显示图片
#     for i in range(len(batch_images)):
#         # print(batch_images[i])
#         # break
#         img = batch_images[i].reshape(28, 28)
#         save_path = f'test/{batch_labels[i]}'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         cv2.imwrite(f'{save_path}/img_{i}.png', img * 255) 
        
#     if index == 10:
#         break


# 读取文件下所有文件夹
# for file in os.listdir('data/'):
#     if os.path.isdir(os.path.join('data/', file)):
#         print(file)

file_list = [f for f in os.listdir('data/') if os.path.isdir(os.path.join('data/', f))]

file_dict = {}

data_list = []

for file in file_list:
    file_dict[file] = [os.path.join(file, f) for f in os.listdir(os.path.join('data/', file)) if os.path.isfile(os.path.join('data/', file, f))]
    data_list.extend(file_dict[file])
    
for key, value in file_dict.items():
    print(key, value[0])

# 将data_list中的数据打乱
import random
random.shuffle(data_list)

# 划分训练集和测试集
train_list = data_list[:int(len(data_list) * 0.8)]
test_list = data_list[int(len(data_list) * 0.8):]

print(len(train_list), len(test_list))
# 保存训练集和测试集
np.savez('data/train_list.npz', train_list=train_list)
np.savez('data/test_list.npz', test_list=test_list)