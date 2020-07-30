import time
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

from class4.nst_utils import *


def gram_matrix(A):
    """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """

    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, tf.transpose(A))
    ### END CODE HERE ###

    return GA


# tf.reset_default_graph()
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
#     GA = gram_matrix(A)
#     print("GA = " + str(GA.eval()))


def compute_content_cost(a_C, a_G):
    """
       计算内容代价的函数

       参数：
           a_C -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像C的内容的激活值。
           a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的内容的激活值。

       返回：
           J_content -- 实数，用上面的公式1计算的值。

       """

    # 获取a_G的维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # print(m, n_H, n_W, n_C)

    # 对a_C与a_G从3维降到2维
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
    # print(a_C_unrolled.shape)

    # 计算内容代价
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.math.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


# print("测试compute_content_cost")
# tf.reset_default_graph()
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_content = compute_content_cost(a_C, a_G)
#     print("J_content = " + str(J_content.eval()))
#
#     test.close()


def compute_layer_style_cost(a_S, a_G):
    """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    # print(a_S.shape)

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))
    # print(GS.shape)

    J_style_layer = tf.reduce_sum(tf.square(tf.math.subtract(GS, GG))) / (4 * tf.square(tf.to_float(n_H * n_W * n_C)))

    return J_style_layer


# tf.reset_default_graph()
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_style_layer = compute_layer_style_cost(a_S, a_G)
#
#     print("J_style_layer = " + str(J_style_layer.eval()))

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)
]


def compute_style_cost(model, STYLE_LAYERS):
    """
        Computes the overall style cost from several chosen layers

        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        # print("a_S = ",a_S)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style

    return J


# tf.reset_default_graph()
# with tf.Session() as test:
#     np.random.seed(3)
#     J_content = np.random.randn()
#     J_style = np.random.randn()
#     J = total_cost(J_content, J_style)
#     print("J = " + str(J))

def model_nn(sess, input_image, num_iterations=200):
    sess.run(tf.global_variables_initializer())
    debug2 = sess.run(model['input'].assign(input_image))
    save_image("output/debug2.jpg", debug2)

    for i in range(num_iterations):
        # save_image("output/" + str(i) + "conv1_1.jpg", sess.run(model['conv1_1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv1_2.jpg", sess.run(model['conv1_2'])[:,:,:,0])
        # save_image("output/" + str(i) + "avgpool1.jpg", sess.run(model['avgpool1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv2_1.jpg", sess.run(model['conv2_1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv2_2.jpg", sess.run(model['conv2_2'])[:,:,:,0])
        # save_image("output/" + str(i) + "avgpool2.jpg", sess.run(model['avgpool2'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv3_1.jpg", sess.run(model['conv3_1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv3_2.jpg", sess.run(model['conv3_2'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv3_3.jpg", sess.run(model['conv3_3'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv3_4.jpg", sess.run(model['conv3_4'])[:,:,:,0])
        # save_image("output/" + str(i) + "avgpool3.jpg", sess.run(model['avgpool3'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv4_1.jpg", sess.run(model['conv4_1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv4_2.jpg", sess.run(model['conv4_2'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv4_3.jpg", sess.run(model['conv4_3'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv4_4.jpg", sess.run(model['conv4_4'])[:,:,:,0])
        # save_image("output/" + str(i) + "avgpool4.jpg", sess.run(model['avgpool4'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv5_1.jpg", sess.run(model['conv5_1'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv5_2.jpg", sess.run(model['conv5_2'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv5_3.jpg", sess.run(model['conv5_3'])[:,:,:,0])
        # save_image("output/" + str(i) + "conv5_4.jpg", sess.run(model['conv5_4'])[:,:,:,0])
        # save_image("output/" + str(i) + "avgpool5.jpg", sess.run(model['avgpool5'])[:,:,:,0])
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        save_image("output/" + str(i) + ".jpg", generated_image)

        print(generated_image[0][12][14])
        # if i % 20 == 0:
        Jt, Jc, Js = sess.run([J, J_content, J_style])
        print("Iteration " + str(i) + ":")
        print("total cost = " + str(Jt))
        print("content cost = " + str(Jc))
        print("style cost = " + str(Js))

    save_image("output/generated_image.jpg", generated_image)

    return generated_image


if __name__ == '__main__':
    print("开启会话")
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())

    print("加载内容图像")
    # content_image = io.imread("images/louvre.jpg")  # (600, 800, 3)
    # content_image = Image.open("images/louvre.jpg")
    content_image = io.imread("images/louvre.jpg")  # shape = (300, 400, 3)
    # plt.imshow(content_image)
    # plt.show()
    print("加载风格图像")
    # style_image = io.imread("images/monet_800600.jpg")  # (600, 800, 3)
    # style_image = Image.open("images/monet_800600.jpg")
    style_image = io.imread("images/monet_800600.jpg")  # (300, 400, 3)
    # plt.imshow(style_image)
    # plt.show()
    print("归一化图像")
    content_image = reshape_and_normalize_image(content_image)  # shape = (1, 300, 400, 3)
    style_image = reshape_and_normalize_image(style_image)  # shape = (1, 300, 400, 3)

    print("生成随机初始图像")
    generated_image = generate_noise_image(content_image)  # shape = (1, 300, 400, 3)
    # plt.imshow(generated_image[0])
    # plt.show()

    print("加载VGG模型")
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    # print(model)

    debug0 = sess.run(model['input'].assign(content_image))
    # print(debug0[0,:,:,0].shape)
    # save_image("output/debug0.jpg", debug0[:, :, :, 0])
    # save_image("output/conv1_1.jpg", sess.run(model['conv1_1'])[:,:,:,0])
    # save_image("output/conv1_2.jpg", sess.run(model['conv1_2'])[:,:,:,0])
    # save_image("output/avgpool1.jpg", sess.run(model['avgpool1'])[:,:,:,0])
    # save_image("output/conv2_1.jpg", sess.run(model['conv2_1'])[:,:,:,0])
    # save_image("output/conv2_2.jpg", sess.run(model['conv2_2'])[:,:,:,0])
    # save_image("output/avgpool2.jpg", sess.run(model['avgpool2'])[:,:,:,0])
    # save_image("output/conv3_1.jpg", sess.run(model['conv3_1'])[:,:,:,0])
    # save_image("output/conv3_2.jpg", sess.run(model['conv3_2'])[:,:,:,0])
    # save_image("output/conv3_3.jpg", sess.run(model['conv3_3'])[:,:,:,0])
    # save_image("output/conv3_4.jpg", sess.run(model['conv3_4'])[:,:,:,0])
    # save_image("output/avgpool3.jpg", sess.run(model['avgpool3'])[:,:,:,0])
    # save_image("output/conv4_1.jpg", sess.run(model['conv4_1'])[:,:,:,0])

    out = model['conv4_2']
    a_C = sess.run(out)

    # save_image("output/conv4_2.jpg", a_C[:,:,:,0])
    # save_image("output/conv4_3.jpg", sess.run(model['conv4_3'])[:,:,:,0])
    # save_image("output/conv4_4.jpg", sess.run(model['conv4_4'])[:,:,:,0])
    # save_image("output/avgpool4.jpg", sess.run(model['avgpool4'])[:,:,:,0])
    # save_image("output/conv5_1.jpg", sess.run(model['conv5_1'])[:,:,:,0])
    # save_image("output/conv5_2.jpg", sess.run(model['conv5_2'])[:,:,:,0])
    # save_image("output/conv5_3.jpg", sess.run(model['conv5_3'])[:,:,:,0])
    # save_image("output/conv5_4.jpg", sess.run(model['conv5_4'])[:,:,:,0])
    # save_image("output/avgpool5.jpg", sess.run(model['avgpool5'])[:,:,:,0])

    # print("a_C = ",a_C)
    a_G = out
    # print("a_G = ",a_G)
    # print(a_C.shape)  # (1, 38, 50, 512)
    # print(a_G.shape)
    J_content = compute_content_cost(a_C, a_G)
    # print(J_content)

    debug1 = sess.run(model['input'].assign(style_image))
    save_image("output/debug1.jpg", debug1)
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, 10, 40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image)
