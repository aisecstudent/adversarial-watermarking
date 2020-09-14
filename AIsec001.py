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
使用说明:python test.py image1_path  trigger_path
例如:python test.py ./image1.png ./image2.jpg

image1_path:干净图片所在路径
trigger_path:目标图片所在路径

环境配置:
python 3.5
tensorflow 1.13
'''

def main(argv):

    #输入异常处理
    if len(argv)!=3:
        print('请按照如下格式调用程序:AIsec001.py image1_path image2_path')
        sys.exit(0)

    print(argv[1],type(argv[1]))
    print(argv[2],type(argv[2]))

    #对干净图片image1进行预处理
    img_path = argv[1]
    img_class = 500
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    wide = img.width > img.height
    new_w = 299 if not wide else int(img.width * 299 / img.height)
    new_h = 299 if wide else int(img.height * 299 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)

    #对目标图片image2进行预处理
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

    #加载inception模型
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

    #加载模型参数
    data_dir = tempfile.mkdtemp()
    inception_tarball, _ = urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

    restore_vars = [
        var for var in tf.global_variables()
        if var.name.startswith('InceptionV3/')]

    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


    #显示图片
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


    #测试image1和image2
    p=classify(img)
    p2=classify(img2)
    #print (np.argmin(p),type(img),np.shape(img))#751 <class 'numpy.ndarray'> (299, 299, 3)
    #print(np.shape(p))#(1000,)

    #初始化生成对抗样本
    x = tf.placeholder(tf.float32, (299, 299, 3))

    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)
    learning_rate = tf.placeholder(tf.float32, ())
    y_tar = tf.placeholder(tf.float32, (1000,))

    #定义优化目标，loss_ssim用于约束人眼对扰动的感知程度
    x_initial = tf.constant(img)
    loss_ssim=(1-tf.reduce_mean(tf.image.ssim(x_hat,x_initial,max_val=1.0)))/2.0
    loss_ssim=tf.reshape(loss_ssim,shape=(1,))
    loss_ssim=tf.square(loss_ssim)

    #loss1用于约束对抗生成的目标软标签
    #在'Hidden Trigger Backdoor Attacks'中，作者使用卷积后提取的特征距离作为优化目标中的一项
    #这里我们简化了这一过程，使用目标样本在最后一层所得到的特征作为对抗生成的目标
    #如果你希望利用对抗样本实现更为复杂的后门攻击，可以参考上述论文的方法，重新自定义模型并训练
    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_tar)
    loss = loss_ssim+loss1

    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

    #投影梯度下降方法生成对抗样本
    epsilon = tf.placeholder(tf.float32, ())

    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    #定义一个非常小的扰动
    demo_epsilon = 2.0 / 255.0
    demo_lr = 0.1
    demo_steps = 200

    #执行初始化步骤
    sess.run(assign_op, feed_dict={x: img})

    for i in range(demo_steps):
        #梯度下降步骤
        _, loss_value , logits_value,ssim_value= sess.run(
            [optim_step, loss ,loss1,loss_ssim],
            feed_dict={learning_rate: demo_lr, y_tar:p2})
        #投影步骤
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i + 1) % 10 == 0:
            print('step %d, loss=%g' % (i + 1, loss_value))

    #测试对抗样本
    adv = x_hat.eval()
    classify(adv)

    #保存生成的对抗样本图片
    imsave('adv_image' + '.png', adv)

if __name__ == "__main__":
    main(sys.argv)
