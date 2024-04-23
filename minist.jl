using TyDeepLearning
using TyPlot
using TyImages
using CSV
using DataFrames
using Random
# using 
# 训练卷积神经网络用于图像分类
function loadMINIST(path)
    df = CSV.read(path * "digitTrain.csv", DataFrame)
    img_name = df.image
    img = Array{UInt8,4}(undef,length(df.image),1,28,28)
    label = df.digit
    for i in 1:length(df.image)
        img[i,1,:,:] = imread(path * img_name[i])
    end
    return img, label
end
# path = TyDeepLearning.dataset_dir("digit")
path = pwd()*"/MINIST/"
println("$path")
img, label = loadMINIST(path)
p = randperm(5000)
index = p[1:20]

figure(1)
for i in eachindex(range(1, 20))
    subplot(4, 5, i)
    imshow(img[index[i], 1, :, :])
end

index1 = p[1:750]
index2 = p[751:end]
X_train = img[index2, :, :, :]
Y_train = label[index2]
X_Test = img[index1, :, :, :]
Y_Test = label[index1]

options = trainingOptions(
    "CrossEntropyLoss", "Adam", "Accuracy", 128, 10, 0.001; Plots=true
)

# layers = SequentialCell([
#     convolution2dLayer(1, 20, 5),
#     reluLayer(),
#     maxPooling2dLayer(2; Stride=2),
#     flattenLayer(),
#     fullyConnectedLayer(20 * 14 * 14, 10),
#     softmaxLayer(),
# ])

layers = SequentialCell([
    convolution2dLayer(1, 8, 3),
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

net = trainNetwork(X_train, Y_train, layers, options)

YPred = TyDeepLearning.predict(net, X_Test)
classes = [i - 1 for i in range(1, 10)]
YPred1 = probability2classes(YPred, classes)
acc = accuracy(YPred, Y_Test)
println(acc)

figure(3)
p2 = randperm(750)
index = p2[1:9]
for i in eachindex(range(1, 9))
    TyPlot.subplot(3, 3, i)
    TyImages.imshow(X_Test[index[i], 1, :, :])
    title1 = "Prediction Label"
    title2 = string(YPred1[index[i]])
    title(string(title1, ": ", title2))
end
