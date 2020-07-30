import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    # xx, yy的大小为((x_max-x_min)/h, (y_max-y_min)/h)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # Z.shape = (1038240,)
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    # Z.shape = (1008, 1030)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), s=5, cmap=plt.cm.Spectral)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        # j = 0, ix = range(0,200)
        # j = 1, ix = range(200,400)
        ix = range(N * j, N * (j + 1))
        # t.shape = (200,)
        # j = 0, t是[0, 3.12]均分200个点，加随机噪声
        # j = 1, t是[3.12, 6.24]均分200个点，加随机噪声
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        # r.shape = (200,)
        # r是每一个点离中心(0, 0)的距离，加随机噪声
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        # X.shape = (400, 2)
        # X是上述点的坐标
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        # Y.shape = (400, 1)
        # Y是上述点的属性，0或者1
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
