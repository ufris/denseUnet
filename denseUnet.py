import tensorflow as tf
from layer import *
import tensorlayer as tl

class Denseunet_Model:
    def __init__(self):
        self.x_data = tf.placeholder(dtype=tf.float32,shape=[None,224,224,1])
        self.y_data = tf.placeholder(dtype=tf.float32,shape=[None,224,224,2])
        self.train_bool = tf.placeholder(dtype=bool)
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.class_num = 2
        self.growth = 24
        self.first_filter = 32

        # result
        self.logit = self.denseunet()

        # softmax
        self.predict = tf.nn.softmax(self.logit)
        # self.aa = tf.get_default_graph().get_tensor_by_name('Softmax:0')
        # print(self.aa)
        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        # cost
        self.cost = self.get_cost()

        # accuracy
        self.accuracy = self.get_accuracy()

    def get_cost(self):
        a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_data,logits=self.predict))
        return a

    def get_accuracy(self):
        a = iou_coe(output=self.predict[:,:,:,:1], target=self.y_data[:,:,:,:1])
        return a

    def denseunet(self):
        def block(in_layer, out_filter, cnt, is_train):
            for i in range(cnt):
                temp_layer = batch_norm(in_layer, is_train)
                temp_layer = tf.nn.leaky_relu(temp_layer)
                temp_layer = ori_conv2(temp_layer, 4 * out_filter, 1, 1, 1, pad='same')
                temp_layer = batch_norm(temp_layer, is_train)
                temp_layer = tf.nn.leaky_relu(temp_layer)
                temp_layer = ori_conv2( temp_layer, out_filter, 3, 3, 1, pad='same')
                in_layer = tf.concat([in_layer, temp_layer], axis=3)
            return in_layer

        def transition(in_layer, out_filter, is_train):
            in_layer = batch_norm(in_layer, is_train)
            in_layer = tf.nn.leaky_relu(in_layer)
            in_layer = ori_conv2(in_layer, out_filter, 1, 1, 1, pad='same')
            in_layer = batch_norm(in_layer, is_train)
            in_layer = tf.nn.leaky_relu(in_layer)
            in_layer = tf.layers.average_pooling2d(in_layer, 2, 2, padding='same')
            return in_layer

        with tf.name_scope(name='down1') as scope:
            down_layer = conv2(self.x_data, self.first_filter, 7, 7, 2, self.train_bool)
            print(down_layer.shape)
            down_layer_pool = tf.layers.max_pooling2d(down_layer, 3, 2, padding='same', name='down1_layer2')
            print(down_layer_pool.shape)

        with tf.name_scope(name='block1') as scope:
            down_layer1 = block(down_layer_pool, self.growth, 4, self.train_bool)
            print(down_layer1.shape)

        with tf.name_scope(name='transition1') as scope:
            down_transition1 = transition(down_layer1, down_layer1.shape[3]//2, self.train_bool)
            print(down_transition1.shape)

        with tf.name_scope(name='block2') as scope:
            down_layer2 = block(down_transition1, self.growth, 8, self.train_bool)
            print(down_layer2.shape)

        with tf.name_scope(name='transition2') as scope:
            down_transition2 = transition(down_layer2, down_layer2.shape[3]//2, self.train_bool)
            print(down_transition2.shape)

        with tf.name_scope(name='block3') as scope:
            down_layer3 = block(down_transition2, self.growth, 24, self.train_bool)
            print(down_layer3.shape)

        with tf.name_scope(name='transition3') as scope:
            down_transition3 = transition(down_layer3, down_layer3.shape[3]//2, self.train_bool)
            print(down_transition3.shape)

        with tf.name_scope(name='block4') as scope:
            down_layer4 = block(down_transition3, self.growth, 16, self.train_bool)
            print(down_layer4.shape)

        with tf.name_scope(name='up4') as scope:
            up_layer3 = up_conv2(down_layer4, down_layer2.shape[3], 2, 2, 2, self.train_bool)
            print(up_layer3.shape)
            up_layer3 = tf.concat([down_layer3, up_layer3], axis=3)
            print(up_layer3.shape)
            up_layer3 = conv2(up_layer3, up_layer3.shape[3], 3, 3, 1, self.train_bool)
            print(up_layer3.shape)

        with tf.name_scope(name='up3') as scope:
            up_layer2 = up_conv2(up_layer3, down_layer1.shape[3], 2, 2, 2, self.train_bool)
            print(up_layer2.shape)
            up_layer2 = tf.concat([down_layer2, up_layer2], axis=3)
            print(up_layer2.shape)
            up_layer2 = conv2(up_layer2, up_layer2.shape[3], 3, 3, 1, self.train_bool)
            print(up_layer2.shape)

        with tf.name_scope(name='up2') as scope:
            up_layer1 = up_conv2(up_layer2, down_layer.shape[3], 2, 2, 2, self.train_bool)
            print(up_layer1.shape)
            up_layer1 = tf.concat([down_layer1, up_layer1], axis=3)
            print(up_layer1.shape)
            up_layer1 = conv2(up_layer1, up_layer1.shape[3], 3, 3, 1, self.train_bool)
            print(up_layer1.shape)

        with tf.name_scope(name='up1') as scope:
            up_layer = up_conv2(up_layer1, down_layer.shape[3], 2, 2, 2, self.train_bool)
            print(up_layer.shape)
            up_layer = tf.concat([down_layer, up_layer], axis=3)
            print(up_layer.shape)
            up_layer = conv2(up_layer, up_layer.shape[3], 3, 3, 1, self.train_bool)
            print(up_layer.shape)

        with tf.name_scope(name='up') as scope:
            up_end = up_conv2(up_layer,int(self.first_filter * (2/3)),2,2,2,self.train_bool)
            print(up_end.shape)
            up_end = conv2(up_end,up_end.shape[3],3,3,1,self.train_bool)
            print(up_end.shape)
            up_end = conv2(up_end, 2, 1, 1, 1, self.train_bool)
            print(up_end.shape)

        return up_end

# self.predict = tf.nn.softmax(self.logits)
#
#         # cost
#         self.cost = self._get_cost(act_func)

if __name__ == '__main__':
    Denseunet_Model()

