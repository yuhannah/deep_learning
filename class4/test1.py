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
    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = tf.reduce_sum(tf.square(tf.math.subtract(a_C_unrolled, a_G_unrolled))) / (
            4 * tf.to_float(n_H * n_W * n_C))

    return J_content


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    J_style_layer = tf.reduce_sum(tf.square(tf.math.subtract(GS, GG))) / (4 * tf.square(tf.to_float(n_H * n_W * n_C)))

    return J_style_layer


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)
]


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha * J_content + beta * J_style

    return J


content_image = io.imread("images/louvre_small.jpg")  # shape = (300, 400, 3)
style_image = io.imread("images/monet.jpg")  # (300, 400, 3)
content_image = reshape_and_normalize_image(content_image)  # shape = (1, 300, 400, 3)
style_image = reshape_and_normalize_image(style_image)  # shape = (1, 300, 400, 3)
generated_image = generate_noise_image(content_image)  # shape = (1, 300, 400, 3)

tf.reset_default_graph()
sess = tf.InteractiveSession()
path = "pretrained-model/imagenet-vgg-verydeep-19.mat"
vgg = scipy.io.loadmat(path)
print(vgg.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'layers', 'meta'])
print(vgg['layers'].shape)  # (1, 43) 43层网络
print(vgg['layers'][0][0].shape)  # (1, 1) [1-2]表示[0][i]第i层网络
# print(vgg['layers'][0][0])
print(vgg['layers'][0][0][0][0][0])  # ['conv1_1']
print(vgg['layers'][0][0][0][0][0][0])  # conv1_1
# print(vgg['layers'][0][0][0][0][0][0][0]) # c
# print(vgg['layers'][0][0][0][0][0][0][1]) # o
# print(vgg['layers'][0][0][0][0][0][0][2]) # n
# print(vgg['layers'][0][0][0][0][0][0][3]) # v
# print(vgg['layers'][0][0][0][0][0][0][4]) # 1
# print(vgg['layers'][0][0][0][0][0][0][5]) # _
# print(vgg['layers'][0][0][0][0][0][0][6]) # 1
# print(vgg['layers'][0][0][0][0][0][0][0][0]) # c
# print(vgg['layers'][0][0][0][0][0][0][0][0][0]) # c
print(vgg['layers'][0][0][0][0][1])  # ['conv']
# print(vgg['layers'][0][0][0][0][1][0]) # conv
# print(vgg['layers'][0][0][0][0][1][0][0]) # c
print(vgg['layers'][0][0][0][0][2].shape)  # (1, 2) [][][0][0][2]表示第i层网络的W和b，[0][0],[0][1]
print(vgg['layers'][0][0][0][0][2][0][0].shape)  # (3, 3, 3, 64)
print(vgg['layers'][0][0][0][0][2][0][1].shape)  # (64, 1)
# print(vgg['layers'][0][0][0][0][3])  # [[ 3  3  3 64]]
# print(vgg['layers'][0][0][0][0][4])  # [[1 1 1 1]]
# print(vgg['layers'][0][0][0][0][5])  # [[1 1]]
# print(vgg['layers'][0][0][0][0][6])  # [[0]]
# print(vgg['layers'][0][0][0][0][7])  # [[1]]
# print(vgg['layers'][0][0][0][0][8])  # []



model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# print(sess.run(model))
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, 10, 40)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, a_C, a_G, num_iterations=200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        out = sess.run(a_G)
        out1, out2, out3 = sess.run([J_content, J_style, J])
        sess.run(train_step)
        out11, out22, out33 = sess.run([J_content, J_style, J])

        generated_image = sess.run(model['input'])
        save_image("output/" + str(i) + ".jpg", generated_image)

        # if i % 20 == 0:
        Jt, Jc, Js = sess.run([J, J_content, J_style])
        print("Iteration " + str(i) + ":")
        print("total cost = " + str(Jt))
        print("content cost = " + str(Jc))
        print("style cost = " + str(Js))

    save_image("output/generated_image.jpg", generated_image)

    return generated_image


# Iteration 0 :
# total cost = 5.05034e+09
# content cost = 7877.66
# style cost = 1.26257e+08
# Iteration 20 :
# total cost = 9.43274e+08
# content cost = 15186.9
# style cost = 2.35781e+07
# Iteration 40 :
# total cost = 4.84897e+08
# content cost = 16785.0
# style cost = 1.21182e+07
# Iteration 60 :
# total cost = 3.12573e+08
# content cost = 17465.8
# style cost = 7.80997e+06
# Iteration 80 :
# total cost = 2.28137e+08
# content cost = 17714.9
# style cost = 5.69899e+06
# Iteration 100 :
# total cost = 1.80694e+08
# content cost = 17895.4
# style cost = 4.51287e+06
# Iteration 120 :
# total cost = 1.49996e+08
# content cost = 18034.3
# style cost = 3.74539e+06
# Iteration 140 :
# total cost = 1.27698e+08
# content cost = 18186.8
# style cost = 3.1879e+06
# Iteration 160 :
# total cost = 1.10698e+08
# content cost = 18354.2
# style cost = 2.76287e+06
# Iteration 180 :
# total cost = 9.73407e+07
# content cost = 18500.9
# style cost = 2.42889e+06

model_nn(sess, generated_image, a_C, a_G)
