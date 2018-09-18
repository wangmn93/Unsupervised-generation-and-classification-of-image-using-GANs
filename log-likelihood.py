import my_utils
import numpy as np
from matplotlib import pyplot as plt
# X,Y = my_utils.loadFullFashion_MNSIT(shift=False)
#X, Y = my_utils.load_data('mnist')

import scipy.io as sio
train_data = sio.loadmat('../train_32x32.mat')
#
X = train_data['X']/255.
Y = train_data['y']
X = X.transpose([3, 0, 1, 2])
test = []
# X = np.reshape(X, [73257,3072])
for x,y in zip(X,Y):
    if y == 1:
        # plt.imshow(x)
        # plt.show()
        test.append(x)
test = np.array(test)
test = np.reshape(test, [len(test),3072])
from sklearn.neighbors import KernelDensity
import time

start = time.time()
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(test[:4000])
end = time.time()
print end - start
# test = X[:2000]
# x_2 = X[20000:30000]
# num_data = 70000


query_image = ['measure/sample-0.jpg','measure/sample-0-0.jpg','measure/final_fake_images2_1.png', 'measure/sample-0-11.jpg']
from PIL import Image
import  utils
for q in query_image:
    t = np.asarray(Image.open(q)) / 255.
    temp = []
    for i in range(8):
        for j in range(8):
            temp.append(t[j * 32:(j + 1) * 32, i * 32:(i + 1) * 32, :])
    temp = np.array(temp)
    temp = np.reshape(temp, [len(temp), 3072])
    # plt.axis('off')
    # f, axarr = plt.subplots(1, 2)
    #
    # axarr[0].imshow(utils.immerge(temp, 8, 8))
    # plt.imshow(t)
    # plt.s



    print q,kde.score(temp)/len(temp)
