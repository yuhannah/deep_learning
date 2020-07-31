import numpy as np
import tensorflow as tf
import pydot
import matplotlib.pyplot as plt
from IPython.display import SVG
from class4.kt_utils import *

import tensorflow.keras.backend as K

K.set_image_data_format('channels_last')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255
# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("预处理数据")
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def HappyModel(input_shape):
    """
        实现一个检测笑容的模型

        参数：
            input_shape - 输入的数据的维度
        返回：
            model - 创建的Keras的模型

    """
    # 定义一个tensor的placeholder，维度为input_shape
    X_input = tf.keras.layers.Input(input_shape)

    # 使用0填充：X_input的周围填充0
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # 对X使用 CONV -> BN -> RELU 块
    X = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), padding="valid", name='conv0')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # 最大值池化层
    X = tf.keras.layers.MaxPooling2D((2, 2), name="max_pool")(X)

    # 降维，矩阵转化为向量 + 全连接层
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


print("测试HappyModel")
# 创建一个模型实体
print(X_train.shape[1:])
# input_shape = X_train.shape[1:] # 输入（64，64，3) ，conv2D输入（3，64，64）
# print(input_shape[2],input_shape[1],input_shape[0])
happy_model = HappyModel(X_train.shape[1:])
# 编译模型
happy_model.compile("adam", "binary_crossentropy", metrics=['accuracy'])
# 训练模型
# 请注意，此操作会花费你大约6-10分钟。
happy_model.fit(X_train, Y_train, batch_size=50, epochs=40)
# 评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))

print("测试图片（smile）")
img_path = 'E:\\Data\\Huan\\deeplearning_ai_books-master\\images\\smile.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
plt.imshow(img)
plt.show()
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.imagenet_utils.preprocess_input(x)
print(happy_model.predict(x))

print("测试图片（angry）")
img_path = 'E:\\Data\Huan\\deeplearning_ai_books-master\\images\\angry1.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
plt.imshow(img)
plt.show()
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.imagenet_utils.preprocess_input(x)
print(happy_model.predict(x))

happy_model.summary()
tf.keras.utils.plot_model(happy_model, to_file='happy_model.png')
SVG(tf.keras.utils.model_to_dot(happy_model).create(prog='dot', format='svg'))
