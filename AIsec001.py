import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import PIL
import numpy as np
from scipy.misc import imsave
import json
import matplotlib.pyplot as plt

'''
instructions：python test.py image1_path  trigger_path
e.g.  python test.py ./image1 ./image2

image1_path:clean image path
trigger_path:target image path

environment configure:
python 3.5
tensorflow 1.13
'''

def main(argv):
    print(argv[1],type(argv[1]))
    print(argv[2],type(argv[2]))

    # deal with image 1,
    img_path = argv[1]
    img_class = 500
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)

    # deal with image 2
    img2_path = argv[2]
    img2 = PIL.Image.open(img2_path)
    img2 = img2.convert("RGB")
    wide2 = img2.width > img2.height
    new_w2 = 299 if not wide2 else int(img2.width * 299 / img2.height)
    new_h2 = 299 if wide2 else int(img2.height * 299 / img2.width)
    img2 = img2.resize((new_w2, new_h2)).crop((0, 0, 299, 299))
    img2 = (np.asarray(img2) / 255.0).astype(np.float32)

    tf.logging.set_verbosity(tf.logging.ERROR)
    sess = tf.InteractiveSession()

    image = tf.Variable(tf.zeros((299, 299, 3)))

    #load inception model
    def inception(image, reuse):
        preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
        arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, _ = nets.inception.inception_v3(
                preprocessed, 1001, is_training=False, reuse=reuse)
            logits = logits[:,1:] # ignore background class
            probs = tf.nn.softmax(logits) # probabilities
        return logits, probs

    logits, probs = inception(image, reuse=False)

    #load model's weight
    data_dir = tempfile.mkdtemp()
    inception_tarball, _ = urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

    restore_vars = [
        var for var in tf.global_variables()
        if var.name.startswith('InceptionV3/')]

    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


    #show the imaget
    imagenet_json, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/imagenet.json')

    with open(imagenet_json) as f:
        imagenet_labels = json.load(f)

    def classify(img, correct_class=None, target_class=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.sca(ax1)
        p = sess.run(probs, feed_dict={image: img})[0]
        ax1.imshow(img)
        fig.sca(ax1)

        topk = list(p.argsort()[-10:][::-1])
        topprobs = p[topk]
        barlist = ax2.bar(range(10), topprobs)

        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk:
            barlist[topk.index(correct_class)].set_color('g')

        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10),[imagenet_labels[i][:15] for i in topk],rotation='vertical')
        fig.subplots_adjust(bottom=0.2)
        plt.show()
        return p


    #test image1 and image2
    p=classify(img)
    p2=classify(img2)
    #print (np.argmin(p),type(img),np.shape(img))#751 <class 'numpy.ndarray'> (299, 299, 3)
    #print(np.shape(p))#(1000,)

    #initial to generate adversarial sample
    x = tf.placeholder(tf.float32, (299, 299, 3))

    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)
    learning_rate = tf.placeholder(tf.float32, ())
    y_tar = tf.placeholder(tf.float32, (1000,))

    #define loss, loss_ssim is uesd to constrain the human eye's perception of the disturbance
    x_initial = tf.constant(img)
    loss_ssim=(1-tf.reduce_mean(tf.image.ssim(x_hat,x_initial,max_val=1.0)))/2.0
    loss_ssim=tf.reshape(loss_ssim,shape=(1,))
    loss_ssim=tf.square(loss_ssim)

    #loss1 is uesd to constrain tha attack target labeil
    #in 'Hidden Trigger Backdoor Attacks', author use the feature's distance as the loss,
    #here we simply achieve a target attack with target sample's feature in the last layer
    #if you want to achieve a more complex backdoor attack using adversarial samples
    #you can read the author's code or train a new model by yourself.
    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_tar)
    loss = loss_ssim+loss1

    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

    #projected gradient descent
    epsilon = tf.placeholder(tf.float32, ())

    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    demo_epsilon = 2.0 / 255.0  # a really small perturbation
    demo_lr = 0.1
    demo_steps = 200
    demo_target = np.argmin(p)  # 最远target

    # initialization step
    sess.run(assign_op, feed_dict={x: img})

    # projected gradient descent
    for i in range(demo_steps):
        # gradient descent step
        _, loss_value , logits_value,ssim_value= sess.run(
            [optim_step, loss ,loss1,loss_ssim],
            feed_dict={learning_rate: demo_lr, y_tar:p2})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i + 1) % 10 == 0:
            print('step %d, loss=%g' % (i + 1, loss_value))

    adv = x_hat.eval()  # retrieve the adversarial example

    classify(adv, correct_class=img_class, target_class=demo_target)

    #save adversarial image
    imsave('adv_image' + '.png', adv)

if __name__ == "__main__":
    main(sys.argv)