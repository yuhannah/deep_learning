import numpy as np
import h5py
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ipython很好用，但是如果在ipython里已经import过的模块修改后需要重新reload就需要这样
# 在执行用户代码前，重新装入软件的扩展和模块。
# %load_ext autoreload
# autoreload 2：装入所有 %aimport 不包含的模块。
# %autoreload 2

np.random.seed(1)


def zero_pad(X, pad):
    """
    把数据集X的图像边界全部使用0来扩充pad个宽度和高度。

    参数：
        X - 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
        pad - 整数，每个图像在垂直和水平维度上的填充量
    返回：
        X_paded - 扩充后的图像数据集，维度为（样本数，图像高度 + 2*pad，图像宽度 + 2*pad，图像通道数）

    """
    X_paded = np.pad(X, (
        (0, 0),  # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),  # 通道数，不填充
                     'constant', constant_values=0)  # 连续一样的值填充

    return X_paded


# print("测试zero_pad")
# x = np.random.randn(4,3,4,2)
# x_paded = zero_pad(x,2)
# #查看信息
# print("x.shape = ", x.shape)
# print("x_paded.shape = ", x_paded.shape)
# print("x = ", x)
# print("x[1,2] = ", x[1,2])
# print("x_paded[1,2] = ", x_paded[1,2])
# #绘制图
# fig, axarr = plt.subplots(1,2) #一行两列
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])
# plt.show()

def conv_single_step(a_slice_prev, W, b):
    """
        在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
        这里切片大小和过滤器大小相同

        参数：
            a_slice_prev - 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
            W - 权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
            b - 偏置参数，包含在了一个矩阵中，维度为（1,1,1）

        返回：
            Z - 在输入数据的片X上卷积滑动窗口（w，b）的结果。
    """
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)

    return Z


# print("测试conv_single_step")
# #这里切片大小和过滤器大小相同
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
# Z = conv_single_step(a_slice_prev,W,b)
# print("Z = " + str(Z))

def conv_forward(A_prev, W, b, hparameters):
    """
        实现卷积函数的前向传播

        参数：
            A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
            W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
            b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
            hparameters - 包含了"stride"与 "pad"的超参数字典。

        返回：
            Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
            cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """
    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape
    # 获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    # 使用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))
    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    # 自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # print("a_slice_prev.shape = ",a_slice_prev.shape)
                    # 执行单步卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    # 数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))
    # 存储一些缓存值，以便于反向传播使用
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)


# print("测试conv_forward")
# A_prev = np.random.randn(2,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad":2,"stride":1}
# Z, cache_conv = conv_forward(A_prev,W,b,hparameters)
# print("Z.shape = ", Z.shape)
# print("Z = ", Z)
# print("np.mean(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])
# print("cache_conv = ", cache_conv)

def pool_forward(A_prev, hparameters, mode="max"):
    """
       实现池化层的前向传播

       参数：
           A_prev - 输入数据，维度为(m, n_H_prev, n_W_prev, n_C_prev)
           hparameters - 包含了 "f" 和 "stride"的超参数字典
           mode - 模式选择【"max" | "average"】

       返回：
           A - 池化层的输出，维度为 (m, n_H, n_W, n_C)
           cache - 存储了一些反向传播需要用到的值，包含了输入和超参数的字典。
    """
    # 获取输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取超参数的信息
    f = hparameters["f"]
    stride = hparameters["stride"]
    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev
    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # 池化完毕，校验数据格式
    assert (A.shape == (m, n_H, n_W, n_C))
    # 校验完毕，开始存储用于反向传播的值
    cache = (A_prev, hparameters)

    return A, cache


# print("测试pool_forward")
# A_prev = np.random.randn(2,4,4,3)
# hparameters = {"f":4,"stride":1}
# A,cache = pool_forward(A_prev,hparameters,mode="max")
# print("mode = max")
# print("A.shape = ", A.shape)
# print("A = ", A)
# A,cache = pool_forward(A_prev,hparameters,mode="average")
# print("mode = average")
# print("A.shape = ", A.shape)
# print("A = ", A)

def conv_backward(dZ, cache):
    """
        实现卷积层的反向传播

        参数：
            dZ - 卷积层的输出Z的梯度，维度为(m, n_H, n_W, n_C)
            cache - 反向传播所需要的参数，conv_forward()的输出之一

        返回：
            dA_prev - 卷积层的输入（A_prev）的梯度值，维度为(m, n_H_prev, n_W_prev, n_C_prev)
            dW - 卷积层的权值的梯度，维度为(f,f,n_C_prev,n_C)
            db - 卷积层的偏置的梯度，维度为（1,1,1,n_C）

    """
    # 获取cache的值
    (A_prev, W, b, hparameters) = cache
    # 获取A_prev的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    print("A_prev.shape = ", A_prev.shape)
    # 获取dZ的基本信息
    (m, n_H, n_W, n_C) = dZ.shape
    print("dZ.shape = ", dZ.shape)
    # 获取权值的基本信息
    (f, f, n_C_prev, n_C) = W.shape
    print("W.shape = ", W.shape)
    # 获取hparameters的值
    pad = hparameters["pad"]
    stride = hparameters["stride"]
    # 初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    print("dA_prev.shape = ", dA_prev.shape)
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    print("dA_prev_pad.shape = ", dA_prev_pad.shape)

    # 现在处理数据
    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        print("da_prev_pad.shape = ", da_prev_pad.shape)
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    print("da_prev_pad[:,:,0] = ", da_prev_pad[:, :, 0])
    print("da_prev_pad[pad:-pad,pad:-pad,0] = ", da_prev_pad[pad:-pad, pad:-pad, 0])
    # 数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)


# print("测试conv_backward")
# #初始化参数
# A_prev = np.random.randn(10,4,4,3)
# print("A_prev[0,:,:,:] = ",A_prev[0,:,:,:])
# W = np.random.randn(2,2,3,8)
# print("W[:,:,:,0]= ",W[:,:,:,0])
# b = np.random.randn(1,1,1,8)
# print("b[:,:,:,0] = ",b[:,:,:,0])
# hparameters = {"pad":2,"stride":1}
# #前向传播
# Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
# #反向传播
# dA,dW,db = conv_backward(Z,cache_conv)
# print("dA_mean = ", np.mean(dA))
# print("dW_mean = ", np.mean(dW))
# print("db_mean = ", np.mean(db))

def create_mask_from_window(x):
    """
       从输入矩阵中创建掩码，以保存最大值的矩阵的位置。

       参数：
           x - 一个维度为(f,f)的矩阵

       返回：
           mask - 包含x的最大值的位置的矩阵
    """
    mask = x == np.max(x)

    return mask


# print("测试create_mask_from_window")
# x = np.random.randn(2,3)
# mask = create_mask_from_window(x)
# print("x = " + str(x))
# print("x = ",x)
# print("mask = " + str(mask))
# print("mask = ",mask)

def distribute_value(dz, shape):
    """
        给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。

        参数：
            dz - 输入的实数
            shape - 元组，两个值，分别为n_H , n_W

        返回：
            a - 已经分配好了值的矩阵，里面的值全部一样。

    """
    # 获取矩阵的大小
    (n_H, n_W) = shape
    # 计算平均值
    average = dz / (n_H * n_W)
    # 填充入矩阵
    a = np.ones(shape) * average

    return a


# print("测试distribute_value")
# dz = 2
# shape = (2,2)
# a = distribute_value(dz,shape)
# print("a = " + str(a))

def pool_backward(dA, cache, mode="max"):
    """
        实现池化层的反向传播

        参数:
            dA - 池化层的输出的梯度，和池化层的输出的维度一样
            cache - 池化层前向传播时所存储的参数。
            mode - 模式选择，【"max" | "average"】

        返回：
            dA_prev - 池化层的输入的梯度，和A_prev的维度相同

    """
    # 获取cache中的值
    (A_prev, hparameters) = cache
    # 获取hparaeters的值
    f = hparameters["f"]
    stride = hparameters["stride"]
    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    # 开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 选择反向传播的计算方式
                    if mode == "max":
                        # 开始切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 创建掩码
                        mask = create_mask_from_window(a_prev_slice)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    # 数据处理完毕，开始验证格式
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


# print("测试pool_backward")
# A_prev = np.random.randn(5,5,3,2)
# print("A_prev = ",A_prev)
# hparameters = {"stride":1,"f":2}
# A,cache = pool_forward(A_prev,hparameters)
# print("A = ",A)
# print("A.shape = ",A.shape)
# dA = np.random.randn(5,4,2,2)
# dA_prev = pool_backward(dA,cache,mode="max")
# print("mode=max")
# print("mean of dA = ",np.mean(dA))
# print("dA_prev[1,1] = ",dA_prev[1,1])
# dA_prev = pool_backward(dA,cache,mode="average")
# print("mode=average")
# print("mean of dA = ",np.mean(dA))
# print("dA_prev[1,1] = ",dA_prev[1,1])

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import class4.cnn_utils
import class2.tf_utils

tf.compat.v1.disable_eager_execution()

# %matplotlib inline
np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = class2.tf_utils.load_dataset()
X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train = class4.cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = class4.cnn_utils.convert_to_one_hot(Y_test_orig, 6).T

# print("查看数据集")
# index = 6
# plt.imshow(X_train_orig[index])
# plt.show()
# print("y = " + str(np.squeeze(Y_train_orig[:,index])))
# print("数据预处理")
# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
       为session创建占位符

       参数：
           n_H0 - 实数，输入图像的高度
           n_W0 - 实数，输入图像的宽度
           n_C0 - 实数，输入的通道数
           n_y  - 实数，分类数

       输出：
           X - 输入数据的占位符，维度为[None, n_H0, n_W0, n_C0]，类型为"float"
           Y - 输入数据的标签的占位符，维度为[None, n_y]，维度为"float"
    """
    X = tf.compat.v1.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_y])

    return X, Y


# print("测试create_placeholders")
# X,Y = create_placeholders(64,64,3,6)
# print("X = " + str(X))
# print("Y = " + str(Y))

def initialize_parameters():
    """
       初始化权值矩阵，这里我们把权值矩阵硬编码：
       W1 : [4, 4, 3, 8]
       W2 : [2, 2, 8, 16]

       返回：
           包含了tensor类型的W1、W2的字典
    """
    tf.compat.v1.set_random_seed(1)

    W1 = tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.compat.v1.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# print("测试initializer_parameters")
# tf.compat.v1.reset_default_graph()
# with tf.compat.v1.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.compat.v1.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
#
#     sess_test.close()

def forward_propagation(X, parameters):
    """
      实现前向传播
      CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

      参数：
          X - 输入数据的placeholder，维度为(输入节点数量，样本数量)
          parameters - 包含了“W1”和“W2”的python字典。

      返回：
          Z3 - 最后一个LINEAR节点的输出

    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A1 = tf.nn.relu(Z1)
    # Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool2d(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    # Conv2d : 步伐：1，填充方式：“SAME”
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A2 = tf.nn.relu(Z2)
    # Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    # 一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    # 全连接层（FC）：使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3


# print("测试forward_propagation")
# tf.compat.v1.reset_default_graph()
# np.random.seed(1)
# with tf.compat.v1.Session() as sess_test:
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#
#     init = tf.compat.v1.global_variables_initializer()
#     sess_test.run(init)
#
#     a =sess_test.run(Z3,{X:np.random.randn(2,64,64,3),Y:np.random.randn(2,6)})
#     print("Z3= " + str(a))
#
#     sess_test.close()

def compute_cost(Z3, Y):
    """
       计算成本
       参数：
           Z3 - 正向传播最后一个LINEAR节点的输出，维度为（6，样本数）。
           Y - 标签向量的placeholder，和Z3的维度相同

       返回：
           cost - 计算后的成本

    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


# print("测试compute_cost")
# tf.compat.v1.reset_default_graph()
# with tf.compat.v1.Session() as sess_test:
#     np.random.seed(1)
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#
#     init = tf.compat.v1.global_variables_initializer()
#     sess_test.run(init)
#     a = sess_test.run(cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
#     print("cost = " + str(a))
#
#     sess_test.close()

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64, print_cost=True,
          isPlot=True):
    """
        使用TensorFlow实现三层的卷积神经网络
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        参数：
            X_train - 训练数据，维度为(None, 64, 64, 3)
            Y_train - 训练数据对应的标签，维度为(None, n_y = 6)
            X_test - 测试数据，维度为(None, 64, 64, 3)
            Y_test - 训练数据对应的标签，维度为(None, n_y = 6)
            learning_rate - 学习率
            num_epochs - 遍历整个数据集的次数
            minibatch_size - 每个小批量数据块的大小
            print_cost - 是否打印成本值，每遍历100次整个数据集打印一次
            isPlot - 是否绘制图谱

        返回：
            train_accuracy - 实数，训练集的准确度
            test_accuracy - 实数，测试集的准确度
            parameters - 学习后的参数
    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)  # 确保你的数据和我一样
    seed = 3  # 指定numpy的随机种子
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    # 为当前维度创建占位符
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算成本
    cost = compute_cost(Z3, Y)
    # 反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就行了
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 全局初始化所有变量
    init = tf.compat.v1.global_variables_initializer()
    # 开始运行
    with tf.compat.v1.Session() as sess:
        # 初始化参数
        sess.run(init)
        # 开始遍历数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)  # 获取数据块的数量
            seed = seed + 1
            minibatches = class4.cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # 对每个数据块进行处理
            for minibatch in minibatches:
                # 选择一个数据块
                (minibatch_X, minibatch_Y) = minibatch
                # 最小化这个数据块的成本
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            # 是否打印成本
            if print_cost:
                # 每5代打印一次
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为： " + str(minibatch_cost))

            # 记录成本
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        # 数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iterations (per tens)")
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()

        # 开始预测数据
        ## 计算当前的预测情况
        predict_op = tf.arg_max(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy = " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuracy))

        return (train_accuracy, test_accuracy, parameters)


print("测试model")
_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)
