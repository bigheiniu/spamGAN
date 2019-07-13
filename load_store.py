import tensorflow as tf

#load model
with tf.Session() as sess:
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen_mle'))
    # saver = tf.train.import_meta_graph('./model/tmp/ckpt-all-base.meta', import_scope="gaen_mle")
    saver1.restore(sess, "./model/tmp/ckpt-all-base")