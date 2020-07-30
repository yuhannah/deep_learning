import numpy as np
import matplotlib.pyplot as plt
from class1.testCases13 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from class1.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# %matplotlib inline #如果你使用用的是Jupyter Notebook的话请取消注释。

np.random.seed(1)  # 设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。

# 加载数据
# X.shape = (2,400)
# Y.shape = (1,400)
X, Y = load_planar_dataset()
print("=========================加载数据=========================")
print("X的维度为: " + str(X.shape))  # X.shape = (2, 400)
print("Y的维度为: " + str(Y.shape))  # X.shape = (1, 400)
print("数据集里面的数据有：" + str(Y.shape[1]) + " 个")  # 400个数据

print("绘制点图：")
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(400), s=40, cmap=plt.cm.Spectral) #绘制散点图
# 上一语句如出现问题，请使用下面的语句：
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=5, cmap=plt.cm.Spectral)  # 绘制散点图
plt.show()

print("=====================用逻辑回归进行预测=====================")
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)  # 学习模型
LR_predictions = clf.predict(X.T)  # 预测结果
print("平均准确性：" + str(clf.score(X.T, Y.T) * 100) + "%")
score = float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100)
print("平均准确性：%d" % score + "%(正确标记的数据点所占的百分比)")
print("绘制逻辑回归图：")
plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 绘制决策边界
plt.title("Logistic Regression")  # 图标题
plt.show()


def layer_sizes(X, Y, n_hide=4):
    """
    参数：
     X - 输入数据集，维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）
     n_hide - 隐藏层的数量

    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
    """
    n_x = X.shape[0]  # 输入层
    n_h = n_hide  # 隐藏层，硬编码为4
    n_y = Y.shape[0]  # 输出层

    return (n_x, n_h, n_y)


# 测试layer_sizes
print("=========================测试layer_sizes=========================")
X_asses, Y_asses = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_asses, Y_asses)
print("输入层的节点数量为: n_x = " + str(n_x))
print("隐藏层的节点数量为: n_h = " + str(n_h))
print("输出层的节点数量为: n_y = " + str(n_y))


def initialize_parameters(n_x, n_h, n_y):
    """
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y - 输出层节点的数量

    返回：
        parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）

    """
    np.random.seed(2)  # 指定一个随机种子，以便你的输出与我们的一样。
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 测试initialize_parameters
print("=========================测试initialize_parameters=========================")
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    参数：
         X - 维度为（n_x，m）的输入数据。
         parameters - 初始化函数（initialize_parameters）的输出

    返回：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
     """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # 前向传播计算A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # 使用断言确保我的数据格式是正确的
    assert (A1.shape == (W1.shape[0], X.shape[1]))
    assert (A2.shape == (W2.shape[0], X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)


# 测试forward_propagation
print("=========================测试forward_propagation=========================")
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print("mean Z1 = " + str(np.mean(cache["Z1"])))
print("mean A1 = " + str(np.mean(cache["A1"])))
print("mean Z2 = " + str(np.mean(cache["Z2"])))
print("mean A2 = " + str(np.mean(cache["A2"])))


def compute_cost(A2, Y, parameters):
    """
    计算方程（6）中给出的交叉熵成本，

    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量

    返回：
         成本 - 交叉熵成本给出方程（13）
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # 计算成本
    epsilon = 1e-5
    logprobs1 = Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon)
    logprobs2 = np.multiply(Y, np.log(A2 + epsilon)) + np.multiply((1 - Y), np.log(1 - A2 + epsilon))
    # print("logprobs1 = " + str(logprobs1))
    # print("logprobs2 = " + str(logprobs2))
    cost = - np.sum(logprobs2) / m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost


# 测试compute_cost
print("=========================测试compute_cost=========================")
A2_assess, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2_assess, Y_assess, parameters)))


def backward_propagation(parameters, cache, X, Y):
    """
    使用上述说明搭建反向传播函数。

    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dZ11 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    # print("dZ1 = " + str(dZ1))
    # print("dZ1 = " + str(dZ11))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    assert (dW2.shape == (A2.shape[0], A1.shape[0]))
    assert (dW1.shape == (A1.shape[0], X.shape[0]))

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# 测试backward_propagation
print("=========================测试backward_propagation=========================")
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用上面给出的梯度下降更新规则更新参数

    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 测试update_parameters
print("=========================测试update_parameters=========================")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

     """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


# 测试predict
print("=========================测试predict=========================")
parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("预测的平均值 = " + str(np.mean(predictions)))


def nn_model(X, Y, n_hide, num_iterations, print_cost=False):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """

    np.random.seed(3)  # 指定随机种子
    (n_x, n_h, n_y) = layer_sizes(X, Y, n_hide)

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if print_cost:
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))
    return parameters


# 测试nn_model
print("=========================测试nn_model=========================")
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, n_hide=4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print("====================用两层神经网络进行预测====================")
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=True)
# 绘制边界
print("绘制逻辑回归图：")
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

print("=============用两层神经网络进行预测(不同隐藏层数量)=============")
# plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5]  # , 20, 50] #隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    # plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    print("绘制逻辑回归图：")
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(n_h))
    plt.show()
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
# plt.show()


# 数据集
print("=======================加载其他数据=========================")
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
X, Y = noisy_circles
print("X.shape = " + str(X.shape))
print("Y.shape = " + str(Y.shape))
X, Y = X.T, Y.reshape(1, Y.shape[0])
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=False)
print("绘制逻辑回归图(noisy_circles)：")
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

X, Y = noisy_moons
print("X.shape = " + str(X.shape))
print("Y.shape = " + str(Y.shape))
X, Y = X.T, Y.reshape(1, Y.shape[0])
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=False)
print("绘制逻辑回归图(noisy_moons)：")
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

X, Y = blobs
print("X.shape = " + str(X.shape))
print("Y.shape = " + str(Y.shape))
Y = Y % 2
X, Y = X.T, Y.reshape(1, Y.shape[0])
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=False)
print("绘制逻辑回归图(blobs)：")
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

X, Y = gaussian_quantiles
print("X.shape = " + str(X.shape))
print("Y.shape = " + str(Y.shape))
X, Y = X.T, Y.reshape(1, Y.shape[0])
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=False)
print("绘制逻辑回归图(gaussian_quantiles)：")
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

X, Y = no_structure  # Y的类型不明确
print("X.shape = " + str(X.shape))
print("Y.shape = " + str(Y.shape))
X, Y = X.T, Y.T
parameters = nn_model(X, Y, n_hide=4, num_iterations=10000, print_cost=False)
print("绘制逻辑回归图(no_structure)：")
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()
predictions = predict(parameters, X)
# print ('平均准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
