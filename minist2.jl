using TyDeepLearning
using Random
using TyPlot
using TyImages

set_backend(:mindspore)
# 加载数据集，划分训练集、测试集
path = TyDeepLearning.dataset_dir("digit")
XTrain, YTrain = DigitDatasetTrainData(path)
p = randperm(5000)
index1 = p[1:1000]
index2 = p[1001:end]
X_train = XTrain[index2, :, :, :]
Y_train = YTrain[index2, :]
X_validation = XTrain[index1, :, :, :]
Y_validation = YTrain[index1, :]


N, Channel, Hight, Width = size(X_train)
# 原始图片展示
p2 = randperm(N)
index = p2[1:20]
figure(1)
for i in eachindex(range(1, 20))
    subplot(4, 5, i)
    imshow(X_train[index[i], 1, :, :])
end

imageAugmenter = imageDataAugmenter(;RandomRotation = 20)
X_train2 = permutedims(X_train, (1, 3, 4, 2))
imagesize = (28, 28)
augimds = augmentedImageDatastore(imagesize, X_train2, Y_train, imageAugmenter)

augimds = permutedims(augimds, (1, 4, 2, 3))
figure(2)
for i in eachindex(range(1, 20))
    subplot(4, 5, i)
    imshow(augimds[index[i], 1, :, :])
end

# 训练参数设置
# 函数api文档地址 https://www.tongyuan.cc/help/SyslabHelp.html#/Doc/TyDeepLearning/Images/TrainNetwork/trainingOptions.html?searchQuery=training
options = trainingOptions("CrossEntropyLoss", "Adam", "Accuracy", 512, 20, 0.001; Shuffle=true , Plots=true)

# 网络设置
# 常见网络层算子接口
# https://www.tongyuan.cc/help/SyslabHelp.html#/Doc/TyDeepLearning/Images.html#%E7%BD%91%E7%BB%9C%E5%B1%82
layers = SequentialCell([
    convolution2dLayer(Channel, 8, 3),
    batchNormalization2dLayer(8),
    reluLayer(),
    maxPooling2dLayer(2; Stride = 2),
    convolution2dLayer(8, 16, 3),
    batchNormalization2dLayer(16),
    reluLayer(),
    maxPooling2dLayer(2; Stride = 2),
    convolution2dLayer(16, 32, 3),
    batchNormalization2dLayer(32),
    reluLayer(),
    flattenLayer(),
    fullyConnectedLayer(32 * 7 * 7, 10),
    softmaxLayer()
    ])

# 模型训练
# https://www.tongyuan.cc/help/SyslabHelp.html#/Doc/TyDeepLearning/Images/TrainNetwork/trainNetwork.html
net = trainNetwork(augimds, Y_train, layers, options)

# 测试
YPred = TyDeepLearning.predict(net, X_validation )
# 分数转换类别
Y_pred = Int32[]
for i in eachindex(range(1,1000))
    temp_pred = YPred[i,:]
    pred = argmax(temp_pred) - 1
    push!(Y_pred, pred)
end

# 准确率验证
Y_validation = reshape(Y_validation, (1000))
acc = accuracy(YPred, Y_validation)
println(acc)

# 预测结果可视化
figure(4)
p2 = randperm(750)
index = p2[1:9]
for i in eachindex(range(1, 9))
    TyPlot.subplot(3, 3, i)
    TyImages.imshow(X_validation[index[i], 1, :, :])
    title1 = "Prediction Label"
    title2 = string(Y_pred[index[i]])
    title(string(title1, ": ", title2))
end