import os
import gzip
import struct
import scipy.io as sio
import numpy as np


def unzip_gz(file_name):
    unzip_name = file_name.replace('.gz', '')
    gz_file = gzip.GzipFile(file_name)
    open(unzip_name, 'w+').write(gz_file.read())
    gz_file.close()


def mnist_load(data_dir, dataset='train', keep = None, shift=True):
    """
    modified from https://gist.github.com/akesling/5358964

    return:
    1. [-1.0, 1.0] float64 images of shape (N * H * W)
    2. int labels of shape (N,)
    3. # of datas

    modified by Mengnan WANG
    extract subset of MNIST
    """

    if dataset is 'train':
        fname_img = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    elif dataset is 'test':
        fname_img = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    if not os.path.exists(fname_img):
        unzip_gz(fname_img + '.gz')
    if not os.path.exists(fname_lbl):
        unzip_gz(fname_lbl + '.gz')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        struct.unpack('>II', flbl.read(8))
        lbls = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack('>IIII', fimg.read(16))
        if shift:
            imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols) / 127.5 - 1
        else:
            imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols) / 255.

    if keep is None:
        return imgs, lbls, len(lbls)
    else:
        X, Y = [], []
        for x, y in zip(imgs, lbls):
            if y in keep:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y, len(Y)

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_mnist():
    imgs, y, num_train_data = mnist_load('dataset/mnist', shift=False)
    imgs_t, y_t, num_train_data_t = mnist_load('dataset/mnist', dataset='test', shift=False)
    imgs.shape = imgs.shape + (1,)
    imgs_t.shape = imgs_t.shape + (1,)

    X = np.concatenate((imgs, imgs_t))
    X = np.reshape(X, [70000, 28, 28, 1])
    y = np.concatenate((y, y_t))
    return X, y

def load_fmnist():
    imgs, y, num_train_data = mnist_load('dataset/fmnist', shift=False)
    imgs_t, y_t, num_train_data_t = mnist_load('dataset/fmnist', dataset='test', shift=False)
    imgs.shape = imgs.shape + (1,)
    imgs_t.shape = imgs_t.shape + (1,)
    return np.concatenate((imgs, imgs_t)), np.concatenate((y, y_t))

def load_svhn():
    train_data = sio.loadmat('dataset/svhn/train_32x32.mat')
    # test_data = sio.loadmat('dataset/svhn/test_32x32.mat')
    X = train_data['X']/255.
    # X_t = test_data['X']/255.
    y = train_data['y']-1
    # y_t = test_data['y']
    X = X.transpose([3, 0, 1, 2])
    # X_t = X_t.transpose([3, 0, 1, 2])
    return X, y

def load_svhn_code():
    train_data = sio.loadmat('dataset/svhn/train_32x32.mat')
    X = np.load('dataset/svhn/svhn-encode-7.npy')
    y = train_data['y']-1
    return X,y

def load_cifar_10():
    img = None
    label = None
    for i in range(1, 5):
        if i == 1:
            batch = unpickle('dataset/cifar-10-batches-py/data_batch_%d' % i)
            img = batch['data']
            label = batch['labels']
        else:
            batch = unpickle('dataset/cifar-10-batches-py/data_batch_%d' % i)
            img = np.concatenate((img, batch['data']))
            label = np.concatenate((label, batch['labels']))
    batch = unpickle('dataset/cifar-10-batches-py/test_batch')
    img = np.concatenate((img, batch['data']))
    img = np.reshape(img, [len(img), 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    label = np.concatenate((label, batch['labels']))

    return img / 255., label

if __name__=='__main__':
    # x,y = load_mnist()
    # x,y = load_fmnist()
    x,y = load_svhn()
    # x,y = load_cifar_10()
    print x.shape
    x,y = load_svhn_code()


