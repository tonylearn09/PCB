import tensorflow as tf
import logging
import sys
import numpy as np
from new_model import Model

FLAGS = tf.app.flags.FLAGS
NUM_LABELS = 1

def load_test_data():
    images = np.load('test_images.npy')
    images_context = np.load('test_images_context.npy')
    labels = np.load('test_targets.npy')
    images -= np.mean(images, axis=0)
    images_context -= np.mean(images_context, axis=0)

    return (images, images_context, labels)

def batch_data(images, images_context, labels, sess, batch_size=10):
    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    images_context_tensor = tf.convert_to_tensor(images_context, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int16)
    input_queue = tf.train.slice_input_producer([images_tensor, images_context_tensor, labels_tensor], num_epochs=1)
    tf.local_variables_initializer().run()

    image_content = input_queue[0]
    images_context_content = input_queue[1]
    labels_content = input_queue[2]
    #print(image_content.shape)

    image_batch, context_batch, label_batch = tf.train.shuffle_batch(
        [image_content, images_context_content, labels_content], batch_size=batch_size, capacity=10000, min_after_dequeue=2000)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, context_batch, label_batch, coord, threads

def evaluate():
    with tf.Graph().as_default():
        images, context, labels = load_test_data()

        logger = logging.getLogger('tensorflow')
        tf.logging.set_verbosity(tf.logging.INFO)

        x_image = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x1')
        x_context = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x2')
        y = tf.placeholder(shape=[None, NUM_LABELS], dtype=tf.float32, name='y')

        model = Model()

        logits = model.inference(x_image, x_context, keep_prob=1.0, is_training=False)
        #logits = model.inference(images.astype(np.float32), context.astype(np.float32), keep_prob=1.0)
        accuracy = model.accuracy(logits, y)
        fnr, fpr = model.fnr_fpr(logits, y)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            #sess.run(tf.initialize_local_variables())
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            image, context, label, coord, threads = batch_data(images, context, labels, sess=sess, batch_size=FLAGS.batch_size)
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                logger.info("restore from the checkpoint {0}".format(ckpt))
            else:
                sys.exit()
            logger.info('=======Start Testing========')
            batch_num = 1
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
            except tf.errors.OutOfRangeError:
                logger.info('==================Test Finished================')
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 128, 'size of training batches')
    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/', 'path to checkpoint directory')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')
    #tf.app.flags.DEFINE_string('test_data', 'data/mnist_test.csv', 'path to test data')

    tf.app.run()
