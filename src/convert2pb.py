# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import graph_util

from config import *
from dataset import pascal_voc, kitti
from utils.util import bbox_transform, Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support PASCAL_VOC or KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeDet/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpointfnm', '/tmp/bichen/logs/squeezeDet/train/model.ckpt-21678',
                            """Checkpoint file name which will be freezen.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('outputfnm', '', """Output *.pb file name""")
tf.app.flags.DEFINE_string('output_node_names', '', """Output *.pb file name""")


def convert():
  """Convert."""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only supports KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+' \
        or FLAGS.net == 'vgg16small', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16ConvDet(mc)
    elif FLAGS.net == 'resnet50':
      mc = kitti_res50_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = ResNet50ConvDet(mc)
    elif FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc)
    elif FLAGS.net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc)
    elif FLAGS.net == 'vgg16small':
      mc = kitti_vgg16_small_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16SmallConvDet(mc)

    saver = tf.train.Saver(model.model_params, reshape=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = "bbox/trimming/bbox,probability/class_idx,probability/score"
    if FLAGS.output_node_names != '':
        output_node_names = FLAGS.output_node_names

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      init = tf.initialize_all_variables()
      sess.run(init)

      print ('Loading check point file {}...'.format(FLAGS.checkpointfnm))
      saver.restore(sess, FLAGS.checkpointfnm)

      output_graph_def = graph_util.convert_variables_to_constants(sess,
            input_graph_def,  output_node_names.split(",")  )

      with tf.gfile.GFile(FLAGS.outputfnm, "wb") as f:
            f.write(output_graph_def.SerializeToString())

      tf.train.write_graph(output_graph_def, FLAGS.checkpointfnm, FLAGS.outputfnm + '.pbtxt', as_text=True)
      tf.train.write_graph(input_graph_def, FLAGS.checkpointfnm, FLAGS.outputfnm + '1.pbtxt', as_text=True)

      print("Output pbtxt file {}".format(FLAGS.outputfnm))
      print("{} ops in the final graph.".format(len(output_graph_def.node)))


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.outputfnm):
    print('Output file {} already exist!'.format(FLAGS.outputfnm))
    return
  convert()


if __name__ == '__main__':
  tf.app.run()
