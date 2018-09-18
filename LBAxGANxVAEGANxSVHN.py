from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import traceback
import tensorflow as tf
import models_mnist as models
import datetime
import my_utils
from functools import partial
import numpy as np
from keras import objectives
from data import load_svhn_code
def experiment():
    """ param """
    epoch = 80
    batch_size = 64
    n_centroid = 10

    gan_type="LearningByAssociation+GAN+VAEGAN+SVHN"
    dir="results/"+gan_type+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    ''' dataset '''

    X, Y = load_svhn_code()





    import utils
    data_pool = utils.MemoryData({'img': X, 'label':Y}, batch_size)



    """ graphs """
    # heads =1
    # generator = partial(models.mgan_gen, heads=1)
    # discriminator = partial(models.mgan_dis, name='d_2')

    # generator = partial(models.ss_generator2, heads=10) #for aegan
    # generator = partial(models.ss_generator3, heads=10) #for vaegan
    #no weight sharing
    def G(z, reuse=True, name="generator", training=True, heads=10, model=None):
        out = []
        with tf.variable_scope(name, reuse=reuse):
            for i in range(heads):
                out += model(z,reuse=reuse, training=training, name=name+str(i))
            return out

    generator = partial(G, heads=10, model = partial(models.ss_generator3, heads=1))  # for vaegan
    # generator = partial(G, heads=10, model = partial(models.ss_generator2, heads=1))  # for ae,aegan
    import models_32x32
    discriminator = partial(models.ss_discriminator2, name='d_2')
    decoder = partial(models_32x32.decoder, name='decoder')
    # encoder = partial(models.cnn_discriminator, out_dim = 10)
    # from cnn_classifier import cnn_classifier
    encoder = partial(models.ss_classifier, keep_prob = 1., name='classifier')
    # encoder = partial(vat.forward, is_training=False, update_batch_stats=False)
    optimizer = tf.train.AdamOptimizer
    #==============================
    # inputs
    real = tf.placeholder(tf.float32, shape=[batch_size, 100])
    real2 = tf.placeholder(tf.float32, shape=[None, 100])
    # with tf.variable_scope("CNN") as scope:
    r_mean = encoder(real2, reuse=False)
    r_p = tf.nn.softmax(r_mean)
        # r_p = r_mean
    predicts = tf.argmax(r_p, axis=1)

    #=====================
    z = tf.random_normal(shape=(batch_size, 100),
                           mean=0, stddev=1, dtype=tf.float32)
    # z =  tf.placeholder(tf.float32, shape=[None, z_dim])
    fake_set = generator(z, reuse=False)
    fake = tf.concat(fake_set, 0)
    # print(fake.shape)
    r_logit = discriminator(real,reuse=False)
    f_logit = discriminator(fake)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit, labels=tf.ones_like(r_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.zeros_like(f_logit)))
    d_loss = d_loss_real + 1.*d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit, labels=tf.ones_like(f_logit)))

    g_loss_ori = tf.identity(g_loss)
    g_loss = 1.*g_loss #weight down real loss

    for i in range(len(fake_set)):
        onehot_labels = tf.one_hot(indices=tf.cast(tf.scalar_mul(i, tf.ones(batch_size)), tf.int32), depth=n_centroid)
        # if i == 0:
        #     f_mean = encoder(fake_set[i],reuse=False)
        # else:
        # with tf.variable_scope("CNN") as scope:
        #     scope.reuse_variables()
        f_mean = encoder(fake_set[i])
        f_p = tf.nn.softmax(f_mean)
            # f_p = f_mean
            # g_loss += .5*cond_entropy(f_p)
        g_loss += 2.*tf.reduce_mean(objectives.categorical_crossentropy(onehot_labels, f_p))
            # g_loss += cat_weight*tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=f_l, onehot_labels=onehot_labels))



    # trainable variables for each network
    T_vars = tf.trainable_variables()
    # en_var = [var for var in T_vars if var.name.startswith('discriminator')]
    # en_var = [var for var in T_vars if var.name.startswith('CNN')]

    en_var = [var for var in T_vars if var.name.startswith('classifier')]
    g_var = [var for var in T_vars if var.name.startswith('generator')]
    dis_var = [var for var in T_vars if var.name.startswith('d_2')]


    #optimizer
    global_step = tf.Variable(0, name='global_step',trainable=False)
    d_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=dis_var)
    g_step = optimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=g_var)
    g_step2 = optimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss_ori, var_list=g_var)
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
    image_sets_ = generator(z, training= False)
    image_sets = []
    for i in range(len(image_sets_)):
        if i == 0:
            image_sets.append(decoder(image_sets_[i],reuse=False))
        else:
            image_sets.append(decoder(image_sets_[i]))
    #todo
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


            if it>30000:
                _, _ = sess.run([d_step, g_step], feed_dict={real: real_ipt})
            else:
                _, _ = sess.run([d_step, g_step2], feed_dict={real: real_ipt})
            if it % 10 == 0:
                summary = sess.run(merged, feed_dict={real: real_ipt})
                writer.add_summary(summary, it)
            if it%1000 == 0:
                i = 0
                # todo
                for f in image_sets:
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
        de_var = [var for var in tf.trainable_variables() if var.name.startswith('decoder')]
        de_saver = tf.train.Saver(var_list=de_var)
        # ckpt = tf.train.get_checkpoint_state('../vat_tf/log/svhnaug/')
        # print("Checkpoints:", ckpt)
        # if ckpt and ckpt.model_checkpoint_path:

            # ae_saver.restore(sess, ckpt.model_checkpoint_path)
        # sess.run(tf.local_variables_initializer())

        # ae_saver.restore(sess, 'results/cnn-classifier-model.ckpt')  # 0.49
        # ae_saver.restore(sess, 'results/cnn-classifier-noshift-model.ckpt')

        # ae_saver.restore(sess, 'results/svhn-encoder-classifier-model.ckpt')#0.89 ae
        # ae_saver.restore(sess, 'results/svhn-encoder-2-classifier-model.ckpt')#0.81 aegan weight 1
        # ae_saver.restore(sess, 'results/svhn-encoder-3-classifier-model.ckpt')  # 0.85 aegan weight 0.5
        # ae_saver.restore(sess, 'results/svhn-encoder-4-classifier-model.ckpt')  # 0.8134 vaegan weight 0.5
        # ae_saver.restore(sess, 'results/svhn-encoder-5-classifier-model.ckpt')  # 0.9 vaegan weight 0.5->mean embed
        # ae_saver.restore(sess, 'results/svhn-encoder-6-classifier-model.ckpt')  # 0.84 aegan weight 0.2
        ae_saver.restore(sess, 'weights/lba-svhn-code/svhn-encoder-7-classifier-model.ckpt')  # 0.92 vaegan weight 0.5->reduce KL->mean embed
        # ae_saver.restore(sess, 'results/cifar-encode-2-classifier-model.ckpt')  # 0.87 vaegan weight 1->mean embed

        # de_saver.restore(sess, 'results/dcgan-ae-20180813-193414/checkpoint/model.ckpt')#ae encoder 1
        # de_saver.restore(sess, 'results/aegan-20180815-185953/checkpoint/model.ckpt')#aegan weight 1->encode2
        # de_saver.restore(sess, 'results/aegan-20180816-094249/checkpoint/model.ckpt')#aeganweight 0.5->encode 3
        # de_saver.restore(sess, 'results/vaegan-20180816-141528/checkpoint/model.ckpt') #vaeganweight 0.5->encode 4
        # de_saver.restore(sess, 'results/aegan-20180823-105228/checkpoint/model.ckpt')  # aeganweight 0.2->encode 6
        de_saver.restore(sess, 'weights/vaegan-svhn/model.ckpt')  # vaeganweight 0.5->reduce KL->mean embed->encode 7

        # de_saver.restore(sess, 'results/vaegan-20180827-154620/checkpoint/model.ckpt')#cifar
        # de_saver.restore(sess, 'results/vaegan-20180827-171731/checkpoint/model.ckpt') #cifar good
        # dist = [0]*10
        predict_y = sess.run(predicts, feed_dict={real2: X[:1000]})
        acc = cluster_acc(predict_y, Y[:1000])
        print('Accuracy of clustering model: ',acc[0])

        gan_train(max_it, 0)

    except Exception, e:
        traceback.print_exc()
    finally:
        import utils
        i = 0
        # todo
        for f in image_sets:
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
    print('Expriment LearningByAssociation + GAN + VAEGAN on SVHN, one experiments')
    print('=====Exp 1=====')
    experiment()

if __name__ == '__main__':

    run()
