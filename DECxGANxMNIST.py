from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from functools import partial
from keras import backend as K
from keras import objectives
from data import load_mnist

def experiment(dec_init):
    """ param """
    epoch = 40
    batch_size = 64
    lr_dec = 0.002
    lr_gan = 0.0002
    beta1 = 0.5
    z_dim = 10
    n_centroid = 10
    original_dim =784

    # n_critic = 1 #
    # n_generator = 1
    gan_type="DEC+GAN+MNIST"
    dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    ''' dataset '''
    X, y = load_mnist()
    import utils
    data_pool = utils.MemoryData({'img': X, 'label':y}, batch_size)


    """ graphs """
    encoder = partial(models.encoder, z_dim = z_dim)
    decoder = models.decoder
    num_heads = 10
    generator = partial(models.generator_m2, heads=num_heads)
    discriminator = models.discriminator2
    # sampleing = models.sampleing
    optimizer = tf.train.AdamOptimizer

    with tf.variable_scope('kmean', reuse=False):
        tf.get_variable("u_p", [n_centroid, z_dim], dtype=tf.float32)

    # inputs
    real = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
    real2 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    # encoder
    z_mean, _ = encoder(real, reuse=False)
    z_mean2, _ = encoder(real2)

    #=====================
    z = tf.random_normal(shape=(batch_size, z_dim),
                           mean=0, stddev=1, dtype=tf.float32)
    fake_set = generator(z, reuse=False)
    fake = tf.concat(fake_set, 0)
    r_logit = discriminator(real,reuse=False)
    f_logit = discriminator(fake)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
    d_loss = d_loss_real + (1./num_heads)*d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))


    def compute_soft_assign(z):
        with tf.variable_scope('kmean', reuse=True):
            theta_p = tf.get_variable('u_p')
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z,dim=1) - theta_p), axis=2) / 1.))
        q **= (1. + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def target_distribution2(q):
        weight = q ** 1.5 / tf.reduce_sum(q, axis=0)
        return tf.transpose(tf.transpose(weight)/ tf.reduce_sum(q, axis=1))

    def KL(P,Q):
        return tf.reduce_sum(P * tf.log(P/Q), [0,1])

    q = compute_soft_assign(z_mean)
    # predicts = tf.argmax(q, axis=1)
    predicts2 = tf.argmax(compute_soft_assign(z_mean2), axis=1)
    # print('soft dist: ',q.shape)
    t = target_distribution2(q)
    # print('target dist: ',t.shape)
    KL_loss = KL(t, q)
    # beta = 0.01
    # KL_recon_loss = beta*KL_loss + recon_loss

    f_logit_set = []
    real_weight = tf.placeholder(tf.float32, shape=[])
    real_weight_init = 1.
    g_loss = 1.*g_loss #weight down real loss
    diversity_weight = tf.get_variable("diversity_term", [10], dtype=tf.float32)
    for i in range(len(fake_set)):
        onehot_labels = tf.one_hot(indices=tf.cast(tf.scalar_mul(i, tf.ones(batch_size)), tf.int32), depth=n_centroid)
        f_m, _ = encoder(fake_set[i])
        f_l = compute_soft_assign(f_m)
        g_loss += 1.*tf.reduce_mean(objectives.categorical_crossentropy(onehot_labels,f_l))

    # trainable variables for each network
    T_vars = tf.trainable_variables()
    en_var = [var for var in T_vars if var.name.startswith('encoder')]
    de_var = [var for var in T_vars if var.name.startswith('decoder')]
    kmean_var = [var for var in T_vars if var.name.startswith('kmean')]

    g_var = [var for var in T_vars if var.name.startswith('generator')]
    dis_var = [var for var in T_vars if var.name.startswith('discriminator')]


    #optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    global_step = tf.Variable(0, name='global_step',trainable=False)
    # ae_step = optimizer(learning_rate=learning_rate).minimize(recon_loss, var_list=en_var+de_var, global_step=global_step)
    kl_step = tf.train.MomentumOptimizer(learning_rate=lr_dec, momentum=0.9).minimize(KL_loss, var_list=kmean_var+en_var)

    d_step = optimizer(learning_rate=lr_gan, beta1=beta1).minimize(d_loss, var_list=dis_var)
    g_step = optimizer(learning_rate=lr_gan, beta1=beta1).minimize(g_loss, var_list=g_var)

    """ train """
    ''' init '''
    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # saver
    saver = tf.train.Saver(max_to_keep=5)
    # summary writer
    # Send summary statistics to TensorBoard
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.image('Real', real, 12)
    # tf.summary.image('Recon', x_hat, 12)


    image_sets = generator(z, training= False)
    for img_set in image_sets:
        tf.summary.image('G_images', img_set, 12)


    merged = tf.summary.merge_all()
    logdir = dir+"/tensorboard"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    print('Tensorboard dir: '+logdir)

    # ''' initialization '''
    sess.run(tf.global_variables_initializer())

    def kmean_init():
        from sklearn.cluster import KMeans



        sample = sess.run(z_mean2, feed_dict={real2:X})
        kmeans = KMeans(n_clusters=n_centroid, n_init=20).fit(sample)

        with tf.variable_scope('kmean', reuse=True):

            u_p = tf.get_variable('u_p')

            return u_p.assign(kmeans.cluster_centers_)







    def training(max_it, it_offset):
        print("DEC iteration: " + str(max_it))
        for it in range(it_offset, it_offset + max_it):
            real_ipt, y = data_pool.batch(['img','label'])
            _ = sess.run([kl_step], feed_dict={real: real_ipt})


    def gan_train(max_it, it_offset):
        print("GAN iteration: " + str(max_it))

        for it in range(it_offset, it_offset + max_it):
            real_ipt, y = data_pool.batch(['img', 'label'])

            _, _ = sess.run([d_step,g_step], feed_dict={real: real_ipt, real_weight: real_weight_init})
            if it % 10 == 0:
                summary = sess.run(merged, feed_dict={real: real_ipt, real_weight: real_weight_init})
                writer.add_summary(summary, it)
            if it%1000 == 0:
                i = 0
                for f in fake_set:
                    sample_imgs = sess.run(f)

                    sample_imgs = sample_imgs * 2. - 1.
                    save_dir = dir + "/sample_imgs"
                    utils.mkdir(save_dir + '/')

                    my_utils.saveSampleImgs(imgs=sample_imgs, full_path=save_dir + "/" + 'sample-%d-%d.jpg' % (i,it), row=8,
                                            column=8)
                    i += 1
    total_it = 0
    try:
        batch_epoch = len(data_pool) // (batch_size)
        max_it = epoch * batch_epoch

        ae_saver = tf.train.Saver(var_list=en_var+de_var)
        # ae_saver.restore(sess, 'results/ae-20180411-193032/checkpoint/model.ckpt')
        ae_saver.restore(sess, dec_init) #ep100 SGD Momentum 0.94
        # ae_saver.restore(sess, 'results/ae-20180412-134727/checkpoint/model.ckpt')  # ep100 0.824
        #ae_saver.restore(sess,'results/ae-20180427-210832/checkpoint/model.ckpt') #0.62

        #dec training
        load_kmean = kmean_init()
        sess.run(load_kmean)
        #train 2 epochs
        training(2*batch_epoch,0)

        #measure dist
        predict_y = sess.run(predicts2, feed_dict={real2: X})
        acc = my_utils.cluster_acc(predict_y, y)
        print('Accuracy of clustering model: ', acc[0])

        #GAN training
        gan_train(max_it, 2*batch_epoch)

    except Exception, e:
        traceback.print_exc()
    finally:
        import utils
        i = 0
        for f in fake_set:
            sample_imgs = sess.run(f)
            sample_imgs = sample_imgs * 2. - 1.
            save_dir = dir + "/sample_imgs"
            utils.mkdir(save_dir + '/')
            my_utils.saveSampleImgs(imgs=sample_imgs, full_path=save_dir + "/" + 'sample-%d.jpg'%i, row=8, column=8)
            i += 1

        # save checkpoint
        save_path = saver.save(sess, dir+"/checkpoint/model.ckpt")
        print("Model saved in path: %s" % save_path)
        print(" [*] Close main session!")
        sess.close()


def run():
    dec_init_1 = 'weights/dec-ae-mnist/ae-1/model.ckpt'#0.95
    dec_init_2 = 'weights/dec-ae-mnist/ae-2/model.ckpt'#0.62
    print('Expriment DEC+GAN on MNIST, two experiments')
    print('=====Exp 1=====')
    experiment(dec_init_1)
    print('=====Exp 2=====')
    experiment(dec_init_2)

if __name__ == '__main__':
    run()