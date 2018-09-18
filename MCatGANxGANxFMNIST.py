from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from data import load_fmnist
from functools import partial
import numpy as np
from keras import objectives

def experiment(init_w):
    """ param """
    epoch = 40
    batch_size = 64
    n_centroid = 10

    gan_type="ModifiedCatGAN+GAN+FMNIST"
    dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    ''' dataset '''
    # data_pool = my_utils.getFullMNISTDatapool(batch_size, shift=False) #range 0 ~ 1
    # data_pool = my_utils.getFullFashion_MNISTDatapool(batch_size, shift=False)
    X,Y = load_fmnist()
    num_data = len(X)
    import utils
    data_pool = utils.MemoryData({'img': X, 'label':Y}, batch_size)




    """ graphs """

    generator = partial(models.generator_m2, heads=10)
    discriminator = partial(models.discriminator2, name='d_2')
    encoder = partial(models.cnn_discriminator, out_dim = 10)
    optimizer = tf.train.AdamOptimizer

    # inputs
    real = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
    real2 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

    r_mean = encoder(real2, reuse=False)
    r_p = tf.nn.softmax(r_mean)
    predicts = tf.argmax(r_p, axis=1)

    #=====================
    z = tf.random_normal(shape=(batch_size, 62),
                           mean=0, stddev=1, dtype=tf.float32)
    # z =  tf.placeholder(tf.float32, shape=[None, z_dim])
    fake_set = generator(z, reuse=False)
    fake = tf.concat(fake_set, 0)
    r_logit = discriminator(real,reuse=False)
    f_logit = discriminator(fake)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
    d_loss = d_loss_real + 0.1*d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))


    g_loss = 10.*g_loss #weight down real loss
    for i in range(len(fake_set)):
        onehot_labels = tf.one_hot(indices=tf.cast(tf.scalar_mul(i, tf.ones(batch_size)), tf.int32), depth=n_centroid)
        f_mean = encoder(fake_set[i])
        f_p = tf.nn.softmax(f_mean)
        # g_loss += .5*cond_entropy(f_p)
        g_loss += tf.reduce_mean(objectives.categorical_crossentropy(onehot_labels, f_p))
        # g_loss += cat_weight*tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f_l, onehot_labels=onehot_labels))

    # trainable variables for each network
    T_vars = tf.trainable_variables()
    en_var = [var for var in T_vars if var.name.startswith('discriminator')]
    # en_var = [var for var in T_vars if var.name.startswith('encoder')]
    g_var = [var for var in T_vars if var.name.startswith('generator')]
    dis_var = [var for var in T_vars if var.name.startswith('d_2')]


    #optimizer
    global_step = tf.Variable(0, name='global_step',trainable=False)
    d_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=dis_var)
    g_step = optimizer(learning_rate=5*0.0002, beta1=0.5).minimize(g_loss, var_list=g_var)
    """ train """
    ''' init '''
    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # saver
    saver = tf.train.Saver(max_to_keep=5)
    # summary writer
    # Send summary statistics to TensorBoard
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)
    image_sets = generator(z, training= False)
    for img_set in image_sets:
        tf.summary.image('G_images', img_set, 12)


    merged = tf.summary.merge_all()
    logdir = dir+"/tensorboard"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    print('Tensorboard dir: '+logdir)

    # ''' initialization '''
    sess.run(tf.global_variables_initializer())


    ''' train '''
    batch_epoch = len(data_pool) // (batch_size)
    max_it = epoch * batch_epoch

    def cluster_acc(Y_pred, Y):
      from sklearn.utils.linear_assignment_ import linear_assignment
      assert Y_pred.size == Y.size
      D = max(Y_pred.max(), Y.max())+1
      w = np.zeros((D,D), dtype=np.int64)
      for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
      ind = linear_assignment(w.max() - w)
      return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

    def gan_train(max_it, it_offset):
        print("GAN iteration: " + str(max_it))
        # total_it = it_offset + max_it
        for it in range(it_offset, it_offset + max_it):
            real_ipt, y = data_pool.batch(['img', 'label'])
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            # z_ipt = np.random.normal(size=[batch_size, z_dim])
            # if it%700 ==0 and it >0:
            #     global cat_weight_init
            #     cat_weight_init = min(0.7, 1.3*cat_weight_init)
            #     print('cat weight', cat_weight_init)
            _, _ = sess.run([d_step,g_step], feed_dict={real: real_ipt})
            if it % 10 == 0:
                summary = sess.run(merged, feed_dict={real: real_ipt})
                writer.add_summary(summary, it)
            if it%1000 == 0:
                import utils
                i = 0
                for f in fake_set:
                    sample_imgs = sess.run(f)
                    # if normalize:
                    #     for i in range(len(sample_imgs)):
                    sample_imgs = sample_imgs * 2. - 1.
                    save_dir = dir + "/sample_imgs"
                    utils.mkdir(save_dir + '/')
                    # for imgs, name in zip(sample_imgs, list_of_names):
                    my_utils.saveSampleImgs(imgs=sample_imgs, full_path=save_dir + "/" + 'sample-%d-%d.jpg' % (i,it), row=8,
                                            column=8)
                    i += 1
    total_it = 0
    try:
        ae_saver = tf.train.Saver(var_list=en_var)
        ae_saver.restore(sess, init_w)  # 0.73

        # dist = [0]*10
        predict_y = sess.run(predicts, feed_dict={real2: X[:2000]})
        acc = cluster_acc(predict_y, Y[:2000])
        print('Accuracy of clustering model: ',acc[0])
        # for i in predict_y:
        #     dist[i] += 1
        # print(np.array(dist)/float(2000))
        gan_train(max_it, 0)

    except Exception, e:
        traceback.print_exc()
    finally:
        import utils
        i = 0
        for f in fake_set:
            sample_imgs = sess.run(f)
            # if normalize:
            #     for i in range(len(sample_imgs)):
            sample_imgs = sample_imgs * 2. - 1.
            save_dir = dir + "/sample_imgs"
            utils.mkdir(save_dir + '/')
            # for imgs, name in zip(sample_imgs, list_of_names):
            my_utils.saveSampleImgs(imgs=sample_imgs, full_path=save_dir + "/" + 'sample-%d.jpg' % i, row=8, column=8)
            i += 1

        # save checkpoint
        save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
        print("Model saved in path: %s" % save_path)
        print(" [*] Close main session!")
        sess.close()

def run():
    print('Experiment Modified CatGAN+GAN on FMNIST, one experiment')
    print('=====Exp 1=====')
    experiment('weights/mcatgan-fmnist/model.ckpt')

if __name__ == '__main__':
    run()

