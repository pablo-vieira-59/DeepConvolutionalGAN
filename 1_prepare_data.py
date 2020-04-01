import numpy as np
import cv2

images_128 = []
images_64 = []
images_32 = []
images_16 = []
images_8 = []
for i in range(1,101):
    img = cv2.imread('train_images/128/' + str(i) + '.png')

    img64 = cv2.resize(img , (64, 128))
    img32 = cv2.resize(img , (32, 64))
    img16 = cv2.resize(img , (16, 32))
    img8 = cv2.resize(img , (8, 16))

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img64 = cv2.cvtColor(img64, cv2.COLOR_RGB2GRAY)
    #img32 = cv2.cvtColor(img32, cv2.COLOR_RGB2GRAY)
    #img16 = cv2.cvtColor(img16, cv2.COLOR_RGB2GRAY)
    #img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

    cv2.imwrite('train_images/64/' + str(i)+'.png', img64)
    cv2.imwrite('train_images/32/' + str(i)+'.png', img32)
    cv2.imwrite('train_images/16/' + str(i)+'.png', img16)
    cv2.imwrite('train_images8/' + str(i)+'.png', img8)

    img = (img - 127.5) / 127.5
    img64 = (img64 - 127.5) / 127.5
    img32 = (img32 - 127.5) / 127.5
    img16 = (img16 - 127.5) / 127.5
    img8 = (img8 - 127.5) / 127.5

    images_128.append(img)
    images_64.append(img64)
    images_32.append(img32)
    images_16.append(img16)
    images_8.append(img8)

np.save('train_images/images_128.npy', images_128)
np.save('train_images/images_64.npy', images_64)
np.save('train_images/images_32.npy', images_32)
np.save('train_images/images_16.npy', images_16)
np.save('train_images/images_8.npy', images_8)