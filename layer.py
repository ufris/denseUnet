import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = 0.000001) # 클수록 더 큰 페널티

def no_batch_conv2(mk_name,input_layer,output_layer,kernel_x,kernel_y,stride_x,pad='same'):
    a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
                         strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
                         activation = None,padding=pad,name=mk_name)
    a = tf.nn.leaky_relu(a)
    return a

# include name
# def conv2(mk_name,input_layer,output_layer,kernel_x,kernel_y,stride_x,is_train,pad='same'):
#     a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
#                          strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
#                          activation = None,padding=pad,name=mk_name)
#     a = tf.layers.batch_normalization(a,momentum=0.9, epsilon=0.000001,training=is_train)
#     a = tf.nn.leaky_relu(a)
#     return a
# # include name
# def ori_conv2(mk_name,input_layer,output_layer,kernel_x,kernel_y,stride_x,pad='same'):
#     a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
#                          strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
#                          activation = None,padding=pad,name=mk_name)
#     return a
#  # include name
# def up_conv2(mk_name,input_layer,output_layer,kernel_x,kernel_y,stride_x,is_train,pad='same'):
#     a = tf.layers.conv2d_transpose(name=mk_name,inputs=input_layer,filters=output_layer,kernel_size=(kernel_x,kernel_y),
#                                    strides=stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
#                                    padding=pad)
#     a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=0.000001, training=is_train)
#     a = tf.nn.leaky_relu(a)
#     return a

def conv2(input_layer,output_layer,kernel_x,kernel_y,stride_x,is_train,pad='same'):
    a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
                         strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
                         activation = None,padding=pad)
    a = tf.layers.batch_normalization(a,momentum=0.9, epsilon=0.000001,training=is_train)
    a = tf.nn.leaky_relu(a)
    return a

def sigmoid_conv2(input_layer,output_layer,kernel_x,kernel_y,stride_x,is_train,pad='same'):
    a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
                         strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
                         activation = None,padding=pad)
    a = tf.layers.batch_normalization(a,momentum=0.9, epsilon=0.000001,training=is_train)
    a = tf.nn.sigmoid(a)
    return a

def ori_conv2(input_layer,output_layer,kernel_x,kernel_y,stride_x,pad='same'):
    a = tf.layers.conv2d(inputs = input_layer, filters = output_layer,kernel_size = (kernel_x,kernel_y),
                         strides = stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
                         activation = None,padding=pad)
    return a

def up_conv2(input_layer,output_layer,kernel_x,kernel_y,stride_x,is_train,pad='same'):
    a = tf.layers.conv2d_transpose(inputs=input_layer,filters=output_layer,kernel_size=(kernel_x,kernel_y),
                                   strides=stride_x,kernel_initializer=initializer,kernel_regularizer=regularizer,
                                   padding=pad)
    a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=0.000001, training=is_train)
    a = tf.nn.leaky_relu(a)
    return a

def s_conv2(mk_name,input_layer,output_layer,kernel_x,kernel_y,stride_x,pad,is_train):
    a = tf.layers.separable_conv2d(inputs=input_layer,filters =output_layer,kernel_size=(kernel_x,kernel_y),
                                   strides = stride_x,
                                   depthwise_initializer=initializer,depthwise_regularizer=regularizer,
                                   pointwise_initializer=initializer,pointwise_regularizer=regularizer,
                                   activation = None,padding=pad,name=mk_name)
    a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=0.000001, training=is_train)
    a = tf.nn.leaky_relu(a)
    return a

def batch_norm(input_layer,is_train):
    layer = tf.layers.batch_normalization(inputs=input_layer,momentum=0.9, epsilon=0.000001,training=is_train)
    return layer

def drop_out(input_layer,prob,is_train):
    a = tf.layers.dropout(inputs=input_layer,rate=prob,training=is_train)
    return a

def iou_coe(output, target, threshold=0.5, axis=[1, 2, 3], smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou