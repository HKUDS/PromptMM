import sys
import numpy as np
import math
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

 
def MDS(D,d):
    DSquare = D ** 2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)#
    #
    eigValSorted_indices = np.argsort(eigVal)
    #
    topd_eigVec = eigVec[:,eigValSorted_indices[:-d-1:-1]] #
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d-1:-1])))
    return X

 
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 
    x = x.reshape((-1,)) 
    x = torch.from_numpy(x)
    return x

def main(argv=None):
    train_set = mnist.MNIST('./mnist_data', train=True, transform=data_tf, download=True)
    train_data = DataLoader(train_set, batch_size=1, shuffle=True)
    for im, label in train_data:
        X = MDS(im.squeeze().reshape((-1, int(math.sqrt(im.shape[1])))).numpy(),2)
        print(X)

if __name__ == "__main__":
    main()