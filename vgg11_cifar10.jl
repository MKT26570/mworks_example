using TyDeepLearning
using DataFrames
using CSV
using Random
using PyCall
using TyImages
using TyPlot

function get_img_rgb(data)
    py"""
    import numpy as np
    def trans(data):
        return np.array(data).transpose((1,2,0))
    """
    return py"trans"(data)
end

function get_dataset(path)
    py"""
    import numpy as np
    import pickle
    import os
    def fuse_data_list(data_list:list):
        labels = []
        data = []
        for t in data_list:
            labels.append(t[b'labels'])
            arrays_shape = t[b'data'].shape
            data_temp = t[b'data'].reshape((10000, 3, 32, 32))
            data.append(data_temp)
        label_fuse = np.concatenate(labels, axis=0)
        # print(label_fuse.shape)
        data_fuse = np.concatenate(data, axis=0)
        # print(data_fuse.shape)
        return (label_fuse, data_fuse)

    def get_cifar(path):
        path_list = []
        data_all = []
        for i in range(1, 6):
            file_name = f"data_batch_{i}"
            data_path  = os.path.join(path, file_name)
            print(data_path)
            with open(data_path, "rb") as f: 
                dict1 = pickle.load(f, encoding='bytes')
                data_all.append(dict1)
        data = fuse_data_list(data_all)
        with open(os.path.join(path, "batches.meta"), "rb") as f:
            classes = pickle.load(f, encoding='bytes')[b'label_names']
            # classes = classes[b'label_names']
        return data
    """
    return py"get_cifar"(path)
end
# envinit()
# 数据集详情：https://www.cs.toronto.edu/~kriz/cifar.html
path = "cifar-10-batches-py"

a0 = get_dataset(path)

X = a0[2]
Y = a0[1]
pic_num, channel, height, weight = size(a0[2][:,:,:,:])
split = 0.005
train_num = Int32(pic_num * split)
test_num = 100
indexs = randperm(pic_num)

train_index = indexs[1:train_num]
test_index = indexs[train_num+1:train_num+test_num]

X_train = X[train_index, :, :, :]
Y_train = Y[train_index]
X_test = X[test_index, :, :, :]
Y_test = Y[test_index]

# X_train2 = permutedims(X_train, (1, 3, 4, 2))

figure(1)
for i in eachindex(range(1, 20))
    subplot(4, 5, i)
    t1 = X_train[i,:,:,:]
    temp_img = get_img_rgb(t1)
    imshow(temp_img)
end
println("start training")
options = trainingOptions("CrossEntropyLoss", "SGD", "Accuracy", 64, 1, 0.001; Shuffle =true , Plots = true)

figure(2)
# ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
layers = SequentialCell([
    convolution2dLayer(channel, 64, 3),
    batchNormalization2dLayer(64),
    reluLayer(),

    convolution2dLayer(64, 128, 3),
    batchNormalization2dLayer(128),
    reluLayer(),

    maxPooling2dLayer(2; Stride = 2),

    convolution2dLayer(128, 256, 3),
    batchNormalization2dLayer(256),
    reluLayer(),

    convolution2dLayer(256, 256, 3),
    batchNormalization2dLayer(256),
    reluLayer(),

    maxPooling2dLayer(2; Stride = 2),

    convolution2dLayer(256, 512, 3),
    batchNormalization2dLayer(512),
    reluLayer(),

    convolution2dLayer(512, 512, 3),
    batchNormalization2dLayer(512),
    reluLayer(),

    maxPooling2dLayer(2; Stride = 2),

    convolution2dLayer(512, 512, 3),
    batchNormalization2dLayer(512),
    reluLayer(),

    convolution2dLayer(512, 512, 3),
    batchNormalization2dLayer(512),
    reluLayer(),

    maxPooling2dLayer(2; Stride = 2),

    averagePooling2dLayer(2; Stride=1),

    flattenLayer(),
    fullyConnectedLayer(512, 10),
    softmaxLayer()
])

net = trainNetwork(X_train, Y_train, layers, options)

Y_pred = TyDeepLearning.predict(net, X_test)
Y_test = reshape(Y_test, (test_num))
acc = accuracy(Y_pred, Y_test)
