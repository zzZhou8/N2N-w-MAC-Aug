import numpy as np
import matplotlib.pyplot as plt


def See_loss(start,epochs_end):
    curve=np.load('./save/loss_{}_epochs.npy'.format(epochs_end))[start:epochs_end]
    print(curve[-1])
    x=range(start,epochs_end)
    plt.plot(x, curve, 'r', lw=1)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train_loss"])

#See_loss(100,23000)

