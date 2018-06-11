from denseUnet import *
import os, cv2
import numpy as np
import tensorflow as tf

model_path = 'D:/test_mha/model'

def sort_key(key):                               # 이미지를 1 2 3 4 순서로 정렬하기 위한 함수
    return int(key[:key.index('.')])

def load_list(test_directory):              # 이미지 로드하는 함수 ex) [path + 1.png, path + 2.png, path + 3.png ...]
    test_list = []            # [D:\BRATS\testing\HGG_LGG\brats_2013_pat0105_1\VSD.Brain.XX.O.MR_Flair.54199\1.png ...]
    file = os.listdir(test_directory)
    file = sorted(file,key=sort_key)
    for i in file:
        test_list.append(test_directory + '/' + i) # test_list 에 저장

    return test_list

test_directory = 'D:/test_mha/image' # 테스트할 이미지가 들어가 있는 폴더
test_img = load_list(test_directory)
print(load_list(test_directory))

# print(load_list()[0:155])

def for_test_preprocess(test_img):          # 위의 test_list에서 이미지 하나씩 불러와 전처리하기 위한 함수
    read_img = cv2.imread(test_img, 0)      # 이미지를 불러옴
    read_img = cv2.resize(read_img, (224, 224), interpolation=cv2.INTER_AREA) # 이미지를 224*244 로 사이즈 변환
    if np.max(read_img) != 0:               # 정규화를 위한 mix_max scaling / x - min(x) / max(x) - min(x)
        read_img = np.divide(np.subtract(read_img, np.min(read_img)),
                             np.subtract(np.max(read_img), np.min(read_img)))
    read_img = np.expand_dims(read_img, axis=2) # 차원을 맞추기 위해 강제로 차원수 늘리기
    read_img = np.expand_dims(read_img, axis=0)
    # print(read_img.shape)
    return read_img

def load_model():
    model = os.listdir(model_path)[-1]
    model = model[:model.index('.ckpt')]
    return model_path + '/' + model + '.ckpt'

print(load_model())
model = Denseunet_Model()

saver = tf.train.Saver() # 저장된 모델을 불러옴
# saver = tf.train.import_meta_graph(img_list())

#print(img_list[0][:img_list[0].index('VSD')])
cnt = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 텐서 그래프를 사용하기 전 그래프 안에 있는 변수를 제거
    saver.restore(sess,load_model()) # 저장된 모델을 사용
    # saver.restore(sess, tf.train.latest_checkpoint())

    # graph = tf.get_default_graph()

    for i in range(len(test_img)):
        if cnt == 155:
            cnt = 0
        cnt += 1
        save_path = test_directory + '/' + 'result'
        # D:\BRATS\testing\HGG_LGG\brats_2013_pat0105_1\result
        test_image = for_test_preprocess(test_img[i])
        feed_dict = {model.x_data: test_image, model.keep_prob: 0, model.train_bool: False}
        predict_img = sess.run(model.predict[:,:,:,:1],feed_dict=feed_dict)

        # 컬러 이미지로 변환하기 위해 강제로 rgb값을 만들어 넣는 부분
        red = predict_img * 255 # 예측한 부분은 1로, 아닌 부분은 0으로 나오기 때문에 1 * 255
        green = np.zeros(shape=predict_img.shape,dtype=np.float32)
        blue = np.zeros(shape=predict_img.shape,dtype=np.float32)

        con_img = np.concatenate((blue,green,red),axis=3) # rgb를 합침
        con_img = np.squeeze(con_img,axis=0)
        con_img = cv2.resize(con_img,(240,240),interpolation=cv2.INTER_AREA)
        con_img = con_img.astype(float)

        print(con_img.shape)

        ori_img = cv2.imread(test_img[i])
        ori_img = ori_img.astype(float)

        save_img = cv2.addWeighted(ori_img,0.5,con_img,0.5,1) # (ori_img(원본 이미지)
                                                              # ,0.5 ,con_img(예측 이미지) ,0.5 ,1)
                                                              # 나머지 파라미터값들은 알파와 감마 등 이미지 합성 시
                                                              # 자연스럽게 해줌 (조정가능)

        if not os.path.exists(save_path): # save_path에 폴더가 없으면 자동 생성
            os.makedirs(save_path)

        cv2.imwrite(save_path + '/' + str(cnt) + '.png',save_img)
        # plt.imshow(con_img)
        # plt.show()