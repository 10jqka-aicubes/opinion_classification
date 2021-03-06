#!/usr/bin/env python3
# coding:utf-8

__author__ = ""

"""
BERT模型文件 ckpt转pb 工具

"""

# import contextlib
import json
import os
from enum import Enum
from termcolor import colored
import sys
from model import modeling
import logging
import tensorflow as tf
import argparse
import pickle
import configure_finetuning
from util import training_utils

import pdb


def set_logger(context, verbose=False):
    if os.name == "nt":  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(levelname)-.1s:" + context + ":[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print("I:%s:%s" % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print("D:%s:%s" % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print("E:%s:%s" % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print("W:%s:%s" % (self.context, msg), flush=True)


def create_classification_model(config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """

    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embedding:
    :return:
    """

    bert_config = training_utils.get_bert_config(config)
    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=config.use_tpu,
        embedding_size=config.embedding_size,
    )

    output_layer = bert_model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    task_name = "multilabel"
    with tf.variable_scope("task_specific/" + task_name):
        logits = tf.layers.dense(output_layer, num_labels)
        probabilities = tf.nn.sigmoid(logits)
        log_softmax = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, log_softmax, probabilities)


def init_predict_var(path):
    num_labels = 2
    label2id = None
    id2label = None
    label2id_file = os.path.join(path, "label2id.pkl")
    if os.path.exists(label2id_file):
        with open(label2id_file, "rb") as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            num_labels = len(label2id.items())
        print("num_labels:%d" % num_labels)
    else:
        print("Can't found %s" % label2id_file)
    return num_labels, label2id, id2label


def optimize_class_model(args, logger=None):
    """
    加载中文分类模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """

    if not logger:
        logger = set_logger(colored("CLASSIFICATION_MODEL, Lodding...", "cyan"), args.verbose)
        pass
    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        if args.model_pb_dir is None:
            tmp_dir = args.model_dir
        else:
            tmp_dir = args.model_pb_dir

        pb_file = os.path.join(tmp_dir, "classification_model.pb")
        if os.path.exists(pb_file):
            print("pb_file exits", pb_file)
            return pb_file

        # 增加 从label2id.pkl中读取num_labels, 这样也可以不用指定num_labels参数； 2019/4/17
        if not args.num_labels:
            num_labels, label2id, id2label = init_predict_var(tmp_dir)
        else:
            num_labels = args.num_labels
        # ---

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), "input_ids")
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), "input_mask")
                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.model_dir, "electra_config.json"))
                # bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))
                # hparams = json.loads(args.hparams)
                # bert_config = configure_finetuning.FinetuningConfig(args.model_name, args.data_dir, **hparams)
                # bert_config = configure_finetuning.FinetuningConfig(model_name='small',data_dir='/data2/classifier/bootdo/models/electra_master_en/data',feature_col=1,tag_col=3 ,raw_data_dir='/data2/classifier/bootdo/data/train/train.1010.5090.4970/result_ELECTRA', train_id=5090, test_id=4970)
                loss, per_example_loss, softmax_pred, sigmoid_pred = create_classification_model(
                    config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=None,
                    labels=None,
                    num_labels=num_labels,
                )

                # pred_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32, name='pred_ids')
                # pred_ids = tf.identity(pred_ids, 'pred_ids')
                # pdb.set_trace()

                multi_pred = tf.identity(sigmoid_pred, "pred_prob")
                saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print(args.model_dir)
                latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
                logger.info("loading... %s " % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                logger.info("freeze...")
                from tensorflow.python.framework import graph_util

                tmp_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ["pred_prob"])
                logger.info("predict cut finished !!!")

        # 存储二进制模型到文件中
        logger.info("write graph to a tmp file: %s" % pb_file)
        with tf.gfile.GFile(pb_file, "wb") as f:
            f.write(tmp_graph.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error("fail to optimize the graph! %s" % e, exc_info=True)


if __name__ == "__main__":
    pass

    """
    bert_model_dir="/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12"
    model_dir="/mnt/sda1/transdat/bert-demo/bert/output/demo7"
    model_pb_dir=model_dir 
    max_seq_len=128
    num_labels=2
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser(description="Trans ckpt file to .pb file")
    parser.add_argument("-bert_model_dir", type=str, required=True, help="chinese google bert model path")
    parser.add_argument("-model_dir", type=str, required=True, help="directory of a pretrained BERT model")
    parser.add_argument(
        "-model_pb_dir", type=str, default=None, help="directory of a pretrained BERT model,default = model_dir"
    )
    parser.add_argument("-max_seq_len", type=int, default=128, help="maximum length of a sequence,default:128")
    parser.add_argument("-num_labels", type=int, default=None, help="length of all labels,default=2")
    parser.add_argument("-verbose", action="store_true", default=False, help="turn on tensorflow logging for debug")

    parser.add_argument("--data_dir", required=True, help="Location of data files (model weights, etc).")
    parser.add_argument("--model_name", required=True, help="The name of the model being fine-tuned.")
    parser.add_argument("--hparams", default="{}", help="JSON dict of model hyperparameters.")

    args = parser.parse_args()

    optimize_class_model(args, logger=None)
