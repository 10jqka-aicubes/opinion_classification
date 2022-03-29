# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config controlling hyperparameters for fine-tuning ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# import tensorflow.compat.v1 as tf
import tensorflow as tf


class FinetuningConfig(object):
    """Fine-tuning hyperparameters."""

    def __init__(
        self,
        train_id,
        test_id,
        data_dir,
        model_name,
        feature_col,
        tag_col,
        raw_data_dir,
        model_dir,
        result_dir,
        pretrain_dir,
        **kwargs
    ):
        # general
        self.model_name = model_name
        self.debug = False  # debug mode for quickly running things
        self.log_examples = False  # print out some train examples for debugging
        self.num_trials = 1  # how many train+eval runs to perform
        self.do_train = True  # train a model
        self.do_eval = False  # evaluate the model
        self.do_test = True  # evaluate on the test set
        self.keep_all_models = True  # if False, only keep the last trial's ckpt
        self.tag_col = tag_col  # 训练平台
        self.feature_col = feature_col  # 训练平台
        self.continue_train = False  # 训练平台
        self.train_id = train_id
        self.test_id = test_id
        self.banlance = False
        self.min_tag_num = 10
        # model
        self.model_size = "small"  # one of "small", "base", or "large"
        self.task_names = ["singlelabel"]  # which tasks to learn
        self.model_dir = model_dir
        self.result_dir = result_dir
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = kwargs["model_hparam_overrides"] if "model_hparam_overrides" in kwargs else {}
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 21128  # number of tokens in the vocabulary
        self.do_lower_case = True

        # training
        self.learning_rate = 3e-4
        self.weight_decay_rate = 0
        self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
        # lr * lr_decay^(depth - max_depth) i.e.,
        # shallower layers have lower learning rates
        self.num_train_epochs = 10  # passes over the dataset during training
        self.warmup_proportion = 0.1  # how much of training to warm up the LR for
        self.save_checkpoints_steps = 2000
        self.iterations_per_loop = 1000
        self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
        # to disc if existing ones are found

        # writing model outputs to disc
        self.write_test_outputs = True  # whether to write test set outputs,
        # currently supported for GLUE + SQuAD 2.0
        self.n_writes_test = 5  # write test set predictions for the first n trials

        # sizing
        self.max_seq_length = 168
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.predict_batch_size = 16
        self.num_label = 27
        self.double_unordered = False  # for tasks like paraphrase where sentence
        # order doesn't matter, train the model on
        # on both sentence orderings for each example
        # for qa tasks
        self.max_query_length = 64  # max tokens in q as opposed to context
        self.doc_stride = 128  # stride when splitting doc into multiple examples
        self.n_best_size = 20  # number of predictions per example to save
        self.max_answer_length = 30  # filter out answers longer than this length
        self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
        self.answerable_uses_start_logits = True  # more advanced answerable
        # classifier using predicted start
        self.answerable_weight = 0.5  # weight for answerability loss
        self.joint_prediction = True  # jointly predict the start and end positions
        # of the answer span
        self.beam_size = 20  # beam size when doing joint predictions
        self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
        # 2.0 test outputs

        # TPU settings
        self.use_tpu = False
        self.num_tpu_cores = 1
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project

        # default locations of data files
        self.data_dir = data_dir
        pretrained_model_dir = os.path.join(data_dir, "models", model_name)
        print("pretrained_model_dir:", pretrained_model_dir)
        self.pretrained_model_dir = pretrained_model_dir
        self.raw = raw_data_dir
        self.raw_data_dir = raw_data_dir
        self.label_file = os.path.join(self.model_dir, "label.tsv")

        # add new param
        if self.task_names[0] == "singlelabel":
            result_data = raw_data_dir

        elif kwargs["task_names"][0] == "multilabel":
            result_data = os.path.join(raw_data_dir, "multilabel")
            if not os.path.exists(result_data):
                os.makedirs(result_data)
        self.vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")
        if not tf.gfile.Exists(self.vocab_file):
            self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
        task_names_str = ",".join(kwargs["task_names"] if "task_names" in kwargs else self.task_names)
        self.init_checkpoint = None if self.debug else pretrained_model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        results_dir = self.result_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_txt = results_dir + task_names_str + "_results.txt"
        self.results_pkl = results_dir + task_names_str + "_results.pkl"
        qa_topdir = results_dir + task_names_str + "_qa"
        self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
        self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
        self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
        # preprocessed = os.path.join(pretrained_model_dir ,"finetuning_tfrecords")
        # if not os.path.exists(preprocessed):
        #   os.mkdir(preprocessed+"./finetuning_tfrecords")
        # preprocessed_data_dir = os.path.join(preprocessed, task_names_str + "_tfrecords")
        # if not os.path.exists(preprocessed_data_dir):
        #   os.mkdir((pretrained_model_dir+"./%s_tfrecords)"%task_names_str))
        # self.preprocessed_data_dir = preprocessed_data_dir
        # update defaults with passed-in hyperparameters
        self.update(kwargs)

        if self.do_train:
            self.process_dir = os.path.join(data_dir, "models", model_name)
            preprocessed_data_dir = os.path.join(
                self.process_dir,
                "finetuning_tfrecords",
                task_names_str + "_tfrecords" + ("-debug" if self.debug else ""),
            )
        else:
            preprocessed_data_dir = os.path.join(
                self.result_dir,
                task_names_str + "_tfrecords" + ("-debug" if self.debug else "") + ("_predict" if self.do_test else ""),
            )

        print("self.do_train:", self.do_train)
        if self.do_train and os.path.exists(preprocessed_data_dir):
            shutil.rmtree(preprocessed_data_dir)
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
        self.preprocessed_data_dir = preprocessed_data_dir
        self.test_predictions = os.path.join(
            pretrained_model_dir, "test_predictions", "{:}_{:}_{:}_predictions.pkl"
        ).format

        # default hyperparameters for single-task models
        if len(self.task_names) == 1:
            task_name = self.task_names[0]
            if task_name == "rte" or task_name == "sts":
                self.num_train_epochs = 20.0
            elif "squad" in task_name or "qa" in task_name:
                self.max_seq_length = 512
                self.num_train_epochs = 2.0
                self.write_distill_outputs = False
                self.write_test_outputs = False
            elif task_name == "chunk":
                self.max_seq_length = 256
            else:
                self.num_train_epochs = 10.0

        # default hyperparameters for different model sizes
        if self.model_size == "large":
            self.learning_rate = 5e-5
            self.layerwise_lr_decay = 0.9
        elif self.model_size == "small":
            self.embedding_size = 128

        # debug-mode settings
        if self.debug:
            self.save_checkpoints_steps = 1000000
            self.use_tfrecords_if_existing = False
            self.num_trials = 1
            self.iterations_per_loop = 1
            self.train_batch_size = 32
            self.num_train_epochs = 3.0
            self.log_examples = True

        # passed-in-arguments override (for example) debug-mode defaults
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
            self.__dict__[k] = v
