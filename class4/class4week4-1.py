import time
import cv2
import os
import sys
import numpy as np
import tensorflow as tf
from IPython.display import SVG

from class4.fr_utils import *
from class4.inception_blocks_v2 import *

from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

K.set_image_data_format('channels_first')

np.set_printoptions(threshold=sys.maxsize)

# 获取模型
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
# 打印模型的总参数数量
print("参数数量：" + str(FRmodel.count_params()))
# ------------用于绘制模型细节，可选--------------#
tf.keras.utils.plot_model(FRmodel, to_file='FRmodel1.png')
SVG(tf.keras.utils.model_to_dot(FRmodel).create(prog='dot', format='svg'))


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
       根据公式（4）实现三元组损失函数

       参数：
           y_true -- true标签，当你在Keras里定义了一个损失函数的时候需要它，但是这里不需要。
           y_pred -- 列表类型，包含了如下参数：
               anchor -- 给定的“anchor”图像的编码，维度为(None,128)
               positive -- “positive”图像的编码，维度为(None,128)
               negative -- “negative”图像的编码，维度为(None,128)
           alpha -- 超参数，阈值

       返回：
           loss -- 实数，损失的值
       """
    # 获取anchor, positive, negative的图像编码
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # 第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.math.subtract(anchor, positive)), axis=-1)

    # 第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.math.subtract(anchor, negative)), axis=-1)

    # 第三步：减去之前的两个距离，然后加上alpha
    basic_loss = tf.add(tf.math.subtract(pos_dist, neg_dist), alpha)

    # 通过取带零的最大值和对训练样本的求和来计算整个公式
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


print("测试triplet_loss")
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

# 开始时间
start_time = time.clock()
# 编译模型
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
# 加载权值
load_weights_from_FaceNet(FRmodel)
# 结束时间
end_time = time.clock()
# 计算时差
minium = end_time - start_time
print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")

print("图像编码")
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


# print(database)

def verify(image_path, identity, database, model):
    """
       对“identity”与“image_path”的编码进行验证。

       参数：
           image_path -- 摄像头的图片。
           identity -- 字符类型，想要验证的人的名字。
           database -- 字典类型，包含了成员的名字信息与对应的编码。
           model -- 在Keras的模型的实例。

       返回：
           dist -- 摄像头的图片与数据库中的图片的编码的差距。
           is_open_door -- boolean,是否该开门。
       """
    # 第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = img_to_encoding(image_path, model)

    # 第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - database[identity])
    print(identity, database[identity])
    # 第三步：判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与 " + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open


# print("验证1")
print(verify("images/camera_0.jpg", "younes", database, FRmodel))
print(verify("images/camera_2.jpg", "kian", database, FRmodel))


def who_is_it(image_path, database, model):
    """
        根据指定的图片来进行人脸识别

        参数：
            images_path -- 图像地址
            database -- 包含了名字与编码的字典
            model -- 在Keras中的模型的实例。

        返回：
            min_dist -- 在数据库中与指定图像最相近的编码。
            identity -- 字符串类型，与min_dist编码相对应的名字。
        """
    # 步骤1：计算指定图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = img_to_encoding(image_path, model)

    # 步骤2 ：找到最相近的编码
    ## 初始化min_dist变量为足够大的数字，这里设置为100
    min_dist = 100

    ## 遍历数据库找到最相近的编码
    for (name, db_enc) in database.items():
        print(name, db_enc)
        ### 计算目标编码与当前数据库编码之间的L2差距。
        dist = np.linalg.norm(encoding - db_enc)

        ### 如果差距小于min_dist，那么就更新名字与编码到identity与min_dist中。
        if dist < min_dist:
            min_dist = dist
            identity = name

    # 判断是否在数据库中
    if min_dist > 0.7:
        print("抱歉，您的信息不在数据库中。")
    else:
        print("姓名" + str(identity) + " 差距：" + str(min_dist))

    return min_dist, identity


print(who_is_it("images/camera_0.jpg", database, FRmodel))
