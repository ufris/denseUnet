import numpy as np
import cv2
import matplotlib.pyplot as plt

# 원본 이미지
test_image = cv2.imread('f:/1.jpg')
plt.title("original image")
plt.imshow(test_image)
plt.show()

# 좌우 반전
flip_left_right_image = np.flip(test_image,axis=1)
plt.title("flip_left_right_image")
plt.imshow(flip_left_right_image)
plt.show()

# tensorflow 로 만들때는 tf.image.flip_left_right(image) 로 하시면 됩니다
# https://www.tensorflow.org/api_docs/python/tf/image/flip_left_right

# 상하 반전
flip_up_down_image = np.flip(test_image,axis=0)
plt.title("flip_up_down_image")
plt.imshow(flip_up_down_image)
plt.show()

# 90도 회전
rot90_imamge = np.rot90(test_image,3)
plt.title("rot90_image")
plt.imshow(rot90_imamge)
plt.show()

# 270도 회전
rot90_imamge = np.rot90(test_image,1)
plt.title("rot270_image")
plt.imshow(rot90_imamge)
plt.show()

# 제가 crop 하는 것을 만들어 봤는데 정해진 비율로 이미지를 잘라서 데이터를 늘리는 방법입니다
# 이렇게 하면 데이터를 늘리는 장점 뿐만 아니라 이미지에 노이지를 줘 학습 시 정확도도 올라갑니다
def crop_image(image,direction=0,rate=4):
    if direction == 0:
        image = image[:-int(image.shape[0]/rate),:-int(image.shape[1]/rate),:]
        return image
    elif direction == 1:
        image = image[:-int(image.shape[0] / rate), int(image.shape[1] / rate):, :]
        return image
    elif direction == 2:
        image = image[int(image.shape[0]/rate):,int(image.shape[1]/rate):,:]
        return image
    elif direction == 3:
        image = image[int(image.shape[0] / rate):, :-int(image.shape[1] / rate), :]
        return image

for i in range(4):
    test_image = cv2.imread('f:/1.jpg')
    test_image = crop_image(test_image,i)
    plt.title("crop" + str(i))
    plt.imshow(test_image)
    plt.show()

