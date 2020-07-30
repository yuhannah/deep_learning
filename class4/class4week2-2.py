import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D, GlobalMaxPooling2D, Add
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# from keras.utils.visualize_util import model_to_dot
# from keras.utils.visualize_util import plot
from keras.initializers import glorot_uniform

import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from class4.resnets_utils import *


def identity_block(X, f, filters, stage, block):
    """
        实现图3的恒等块

        参数：
            X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_C_prev )
            f - 整数，指定主路径中间的CONV窗口的维度
            filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
            stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
            block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。

        返回：
            X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)

    """
    # 定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 获取过滤器
    F1, F2, F3 = filters

    # 保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X

    # 主路径的第一部分
    ##卷积层
    # X = Conv2D(F1, 1, 1, border_mode="valid", subsample=(1, 1), name=conv_name_base + "2a")(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    # 主路径的第二部分
    ##卷积层
    # X = Conv2D(F2, f, f, border_mode="same", subsample=(1, 1), name=conv_name_base + "2b")(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    # 主路径的第三部分
    ##卷积层
    # X = Conv2D(F3, 1, 1, border_mode="valid", subsample=(1, 1), name=conv_name_base + "2c")(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    ##没有ReLU激活函数

    # 最后一步：
    ##将捷径与输入加在一起
    X = Add()([X, X_shortcut])

    ##使用ReLU激活函数
    X = Activation("relu")(X)

    return X


print("测试identity_block")
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block="a")

    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
    print("out[0].shape = ", out[0].shape)

    test.close()


def convolutional_block(X, f, filters, stage, block, s=2):
    """
        实现图5的卷积块

        参数：
            X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
            f - 整数，指定主路径中间的CONV窗口的维度
            filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
            stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
            block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
            s - 整数，指定要使用的步幅

        返回：
            X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
    """
    # 定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 获取过滤器数量
    F1, F2, F3 = filters

    # 保存输入数据
    X_shortcut = X

    # 主路径
    ##主路径第一部分
    # X = Conv2D(F1, 1, 1, border_mode='valid', subsample=(s, s))(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    ##主路径第二部分
    # X = Conv2D(F2, f, f, border_mode='same', subsample=(1, 1))(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    ##主路径第三部分
    # print(X[0].shape)
    # X = Conv2D(F3, 1, 1, border_mode='valid', subsample=(1, 1))(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    # print(X[0].shape)

    # 捷径
    # print(X_shortcut[0].shape)
    # X_shortcut = Conv2D(F3, 1, 1, border_mode='valid', subsample=(s, s))(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                        name=conv_name_base + "1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)
    # print(X_shortcut[0].shape)

    # 最后一步
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


print("测试convolutional_block")
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)

    A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block="a")
    test.run(tf.global_variables_initializer())

    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
    print("out[0].shape = ", out[0].shape)

    test.close()


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
     实现ResNet50
     CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
     -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

     参数：
         input_shape - 图像数据集的维度
         classes - 整数，分类数

     返回：
         model - Keras框架的模型

     """

    # 定义tensor类型的输入数据
    X_input = Input(input_shape)

    # 0填充
    X = ZeroPadding2D((3, 3))(X_input)

    # stage1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1", kernel_initializer=glorot_uniform(seed=0))(
        X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b", )
    X = identity_block(X, f=3, filters=[63, 63, 256], stage=2, block="c")

    # stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # stage5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 均值池化层
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # 输出层
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc" + str(classes))(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


print("测试ResNet50")
model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# print("数据预处理")
# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

model.fit(X_train, Y_train, batch_size=32, epochs=2)
preds = model.evaluate(X_test, Y_test)
print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))

print("测试模型")
# 加载模型
model = load_model("ResNet50.h5")
preds = model.evaluate(X_test, Y_test)
print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))

# print("测试图片")
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt # plt 用于显示图片
#
# img_path = 'images/fingers_big/2.jpg'
#
# my_image = image.load_img(img_path, target_size=(64, 64))
# my_image = image.img_to_array(my_image)
#
# my_image = np.expand_dims(my_image,axis=0)
# my_image = preprocess_input(my_image)
#
# print("my_image.shape = " + str(my_image.shape))
# print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
# print(model.predict(my_image))
#
# my_image = scipy.misc.imread(img_path)
# plt.imshow(my_image)
# plt.show()
