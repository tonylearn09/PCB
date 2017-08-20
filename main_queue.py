import numpy as np
import tensorflow as tf
import logging
import os
import math
import time

from new_model import Model

FLAGS = tf.app.flags.FLAGS
NUM_LABELS = 1

def load_train_data():
    images = np.load('train_images.npy')
    images_context = np.load('train_images_context.npy')
    labels = np.load('train_targets.npy')
    images -= np.mean(images, axis=0)
    images_context -= np.mean(images_context, axis=0)

    num_examples = len(images)
    train_num_examples = int(num_examples * 0.8)
    idx = np.arange(0, len(images))
    np.random.shuffle(idx)
    id_train = idx[:train_num_examples]
    id_val = idx[train_num_examples:]
    train_images = images[id_train]
    val_images = images[id_val]
    train_images_context = images_context[id_train]
    val_images_context = images_context[id_val]
    train_labels = labels[id_train]
    val_labels = labels[id_val]
    return (train_images, train_images_context, train_labels, val_images, val_images_context, val_labels)

def preprocess(images, context, seed):
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images, seed=seed)
        context = tf.image.random_flip_up_down(context, seed=seed)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images, seed=seed)
        context = tf.image.random_flip_left_right(context, seed=seed)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3, seed=seed)
        context = tf.image.random_brightness(context, max_delta=0.3, seed=seed)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2, seed=seed)
        context = tf.image.random_contrast(context, 0.8, 1.2, seed=seed)
    if FLAGS.random_rotate:
        #radian = tf.random_uniform([1], maxval=2*math.pi)
        radian = tf.random_uniform([1], minval=-0.174, maxval=0.174)
        images = tf.contrib.image.rotate(images, radian) 
        context = tf.contrib.image.rotate(context, radian) 

    return images, context

def batch_data(images, images_context, labels, sess, batch_size=10):
    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    images_context_tensor = tf.convert_to_tensor(images_context, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int16)
    input_queue = tf.train.slice_input_producer([images_tensor, images_context_tensor, labels_tensor])

    image_content = input_queue[0]
    images_context_content = input_queue[1]
    labels_content = input_queue[2]

    #print(image_content.shape)

    my_seed = np.random.randint(0, 2 ** 31 - 1)
    image_post, context_post = preprocess(image_content, images_context_content, my_seed)

    image_batch, context_batch, label_batch = tf.train.shuffle_batch(
        [image_post, context_post, labels_content], batch_size=batch_size, capacity=10000, min_after_dequeue=2000)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, context_batch, label_batch, coord, threads

def train(images, images_context, labels):
    model = Model()

    with tf.Graph().as_default():
        logger = logging.getLogger('tensorflow')
        tf.logging.set_verbosity(tf.logging.INFO)

        x_image = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x1')
        x_context = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x2')
        y = tf.placeholder(shape=[None, NUM_LABELS], dtype=tf.float32, name='y')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.contrib.framework.get_or_create_global_step()

        logits = model.inference(x_image, x_context, keep_prob=keep_prob,
                                 is_training=is_training)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            image, context, label, coord, threads = batch_data(images, images_context, labels, sess=sess, batch_size=FLAGS.batch_size)
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)

            start_step = 0
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('resotre from the checkpoint {0}'.format(ckpt))
                    start_step += int(ckpt.split('-')[-1])
            logger.info('---Training start---')
            try:
                while not coord.should_stop():
                    start_time=time.time()
                    _, cur_loss, summary, step = sess.run([train_op, loss, summary_op, global_step],
                                                          feed_dict={x_image: image.eval(), x_context: context.eval(), y: label.eval(), keep_prob: 0.5, is_training: True})
                    writer.add_summary(summary, step)
                    #print(step, cur_loss)
                    end_time = time.time()
                    logger.info("the step {0} takes {1}, loss {2}".format(step, end_time-start_time, cur_loss))
                    if step > FLAGS.max_steps:
                        break
                    if step % FLAGS.eval_steps == 1:
                        accuracy_val, step = sess.run([accuracy, global_step],
                                                      feed_dict={x_image: image.eval(), x_context: context.eval(), y: label.eval(), keep_prob: 1.0, is_training: False})
                        logger.info('===============Eval a batch in Train data=======================')
                        logger.info( 'the step {0}: accuracy {1}'.format(step, accuracy_val))
                        logger.info('===============Eval a batch in Train data=======================')
                    if step % FLAGS.save_steps == 1:
                        logger.info('Save the ckpt of {0}'.format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=global_step)
            except tf.errors.OutOfRangeError:
                # print "============train finished========="
                logger.info('==================Train Finished================')
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=global_step)
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


def validation(images, images_context, labels):

    max_batch_num = images.shape[0] // FLAGS.batch_size
    with tf.Graph().as_default():
        model = Model()
        logger = logging.getLogger('tensorflow')
        tf.logging.set_verbosity(tf.logging.INFO)

        x_image = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x1')
        x_context = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x2')
        y = tf.placeholder(shape=[None, NUM_LABELS], dtype=tf.float32, name='y')

        logits = model.inference(x_image, x_context, keep_prob=1.0, is_training=True)
        accuracy = model.accuracy(logits, y)
        fnr, fpr = model.fnr_fpr(logits, y)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            image, context, label, coord, threads = batch_data(images, images_context, labels, sess=sess, batch_size=FLAGS.batch_size)
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()


            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                logger.info("restore from the checkpoint {0}".format(ckpt))
            else:
                sys.exit()

            logger.info('---Validation start---')
            batch_num = 0
            try:
                while not coord.should_stop():
                    logger.info('======Start Batch {0}========'.format(batch_num))
                    total_accuracy, total_fnr, total_fpr, summary = sess.run([accuracy, fnr, fpr, summary_op], feed_dict={
                        x_image: image.eval(), x_context: context.eval(), y: label.eval()})
                    logger.info('Test accuracy: {0}'.format(total_accuracy))
                    logger.info('False negative rate: {0}'.format(total_fnr))
                    logger.info('False positive rate: {0}'.format(total_fpr))
                    logger.info('======End Batch {0}========'.format(batch_num))
                    writer.add_summary(summary, batch_num)
                    batch_num += 1

                    if batch_num > max_batch_num:
                        break
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


def main(argv=None):
    (images, images_context, labels, val_images, val_images_context, val_labels) = load_train_data()
    train(images, images_context, labels)
    #validation(val_images, val_images_context, val_labels)


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 20, 'size of training batches')
    tf.app.flags.DEFINE_integer('max_steps', 200, 'number of training iterations')
    tf.app.flags.DEFINE_integer('eval_steps', 100, 'number of iterations to evaluate')
    tf.app.flags.DEFINE_integer('save_steps', 100, 'number of iterations to save')
    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/', 'path to checkpoint directory')
    #tf.app.flags.DEFINE_string('train_data', 'data/mnist_train.csv', 'path to train and test data')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')
    tf.app.flags.DEFINE_boolean('random_flip_up_down', True, 'Whether to flip_up_down')
    tf.app.flags.DEFINE_boolean('random_flip_left_right', True, 'Whether to flip_left_right')
    tf.app.flags.DEFINE_boolean('random_brightness', True, 'Whether to change brightness')
    tf.app.flags.DEFINE_boolean('random_contrast', True, 'Whether to change Contrast')
    tf.app.flags.DEFINE_boolean('random_rotate', True, 'Whether to rotate')
    tf.app.flags.DEFINE_boolean('restore', False, 'Whether to restore checkpoint')


    tf.app.run()
