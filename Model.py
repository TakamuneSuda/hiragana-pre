
# coding: utf-8

# In[ ]:


import tensorflow as tf
import logging


# In[ ]:


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    
#     入力層
    input_layer = tf.reshape(features['x'], [-1, 32, 32, 1])

#     畳み込み層1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv1')

#     畳み込み層2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv2')

#     プーリング層1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool1')

#     畳み込み層3
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv3')

#     畳み込み層4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv4')

    for i in range(1, 5):
        tf.summary.histogram(
            'layer{}/bias'.format(i),
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                'conv{}/bias'.format(i))[0])
        tf.summary.histogram(
            'layer{}/weights'.format(i),
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                'conv{}/kernel'.format(i))[0])

#         プーリング層2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=2,
        name='pool2')

    
#     全結合層
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    dense = tf.layers.dense(
                inputs=pool2_flat,
                units=1024,
                activation=tf.nn.relu,
                name='dense')

#     50%のニューロンを削除
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name='dropout')

#     出力層
    logits = tf.layers.dense(
        inputs=dropout,
        units=75,
        name='logits')

#     予測の生成
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

#     予測の設定(結果をEstimatorSpecオブジェクトに変換し返す)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#     損失の計算
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#     訓練の設定(訓練結果をEstimatorSpecオブジェクトに変換し返す。)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                   mode=mode, loss=loss, train_op=train_op)

#     評価指標を追加
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

