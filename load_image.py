import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import scipy.misc as sm
import tensorflow as tf

# model = Denseunet_Model() # 모델 그래프

test_path = 'D:/test_mha/VSD.Brain.XX.O.MR_Flair.54193.mha'            # 테스트할 이미지 폴더

def mha_to_png_for_save(test_path): # mha 파일을 png로 바꾸는 함수
    test_mha = os.listdir(test_path)
    sava_mha_path = test_path + '/' + test_mha[0]
    img = io.imread(os.path.join(test_path,test_mha[0]), plugin='simpleitk')
    if not os.path.exists(sava_mha_path):
        os.makedirs(sava_mha_path)
    for i in range(len(img)):
        sm.imsave(sava_mha_path + '\\' + str(i + 1) + '.png', img[i])

# mha_to_png_for_save()