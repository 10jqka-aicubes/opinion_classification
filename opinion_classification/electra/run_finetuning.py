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

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import tensorflow as tf
import time

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils
import numpy as np


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class StatisticHelper(object):
    def __init__(self, name="train"):
        self._name = name
        self._node = {"all": 0, "right": 0}
        self._sentence = {"all": 0, "right": 0}

    def cntRightNode(self):
        self._node["all"] += 1
        self._node["right"] += 1

    def cntWrongNode(self):
        self._node["all"] += 1

    def cntRightSen(self):
        self._sentence["all"] += 1
        self._sentence["right"] += 1

    def cntWrongSen(self):
        self._sentence["all"] += 1

    def getAcc(self):
        return self._sentence["right"] * 100 / (self._sentence["all"] + 0.01)

    def toString(self):
        if self._node["all"] > 0:
            ret = "node: %d/%d = %.2f%%;sen:%d/%d = %.2f%%" % (
                self._node["right"],
                self._node["all"],
                self._node["right"] * 100 / (self._node["all"] + 0.01),
                self._sentence["right"],
                self._sentence["all"],
                self._sentence["right"] * 100 / (self._sentence["all"] + 0.01),
            )
        else:
            ret = "sen:%d/%d = %.2f%%" % (
                self._sentence["right"],
                self._sentence["all"],
                self._sentence["right"] * 100 / (self._sentence["all"] + 0.01),
            )
        return ret


class TagStatisticHelper(object):
    def __init__(self, name="train_tag_"):
        self._name = str(name)
        self._TP = 0
        self._FP = 0
        self._FN = 0
        self._TN = 0

    def cntTPSen(self):
        self._TP += 1

    def cntFPSen(self):
        self._FP += 1

    def cntFNSen(self):
        self._FN += 1

    def cntTNSen(self):
        self._TN += 1

    def getPrecisionAndRecall(self):
        precision = self._TP / (self._TP + self._FP + 0.00001)
        recall = self._TP / (self._TP + self._FN + 0.00001)
        return precision, recall

    def toString(self):
        precision = self._TP / (self._TP + self._FP + 0.0001)
        recall = self._TP / (self._TP + self._FN + 0.0001)
        F_1 = 2 * precision * recall / (precision + recall + 0.0001)
        ret = "name: %s precison: %d/%d = %.2f%%, recall: %d/%d = %.2f%%, F_1: %.2f%%" % (
            self._name,
            self._TP,
            self._TP + self._FP,
            precision * 100,
            self._TP,
            self._TP + self._FN,
            recall * 100,
            F_1 * 100,
        )
        return ret


class FinetuningModel(object):
    """Finetuning model with support for multi-task training."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks, is_training, features, num_train_steps):
        # Create a shared transformer encoder
        bert_config = training_utils.get_bert_config(config)
        self.bert_config = bert_config
        if config.debug:
            bert_config.num_hidden_layers = 3
            bert_config.hidden_size = 144
            bert_config.intermediate_size = 144 * 4
            bert_config.num_attention_heads = 4
        assert config.max_seq_length <= bert_config.max_position_embeddings

        bert_model = modeling.BertModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=features["input_ids"],
            input_mask=features["input_mask"],
            token_type_ids=features["segment_ids"],
            use_one_hot_embeddings=config.use_tpu,
            embedding_size=config.embedding_size,
        )
        percent_done = tf.cast(tf.train.get_or_create_global_step(), tf.float32) / tf.cast(num_train_steps, tf.float32)

        # Add specific tasks
        self.outputs = {"task_id": features["task_id"]}

        losses = []
        for task in tasks:
            with tf.variable_scope("task_specific/" + task.name):
                task_losses, task_outputs = task.get_prediction_module(bert_model, features, is_training, percent_done)
                losses.append(task_losses)
                self.outputs[task.name] = task_outputs
        self.loss = tf.reduce_sum(tf.stack(losses, -1) * tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks, num_train_steps, pretraining_config=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""
        utils.log("Building model...")
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        model = FinetuningModel(config, tasks, is_training, features, num_train_steps)

        # Load pre-trained weights from checkpoint
        init_checkpoint = config.init_checkpoint
        if pretraining_config is not None:
            init_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
            utils.log("Using checkpoint", init_checkpoint)
        tvars = tf.trainable_variables()

        # calculate total number of params
        num_params = sum([np.prod(v.shape) for v in tvars])
        utils.log("##### params: {} #####".format(num_params))

        scaffold_fn = None
        if init_checkpoint:
            assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if config.use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Build model for training or prediction
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                model.loss,
                config.learning_rate,
                num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_proportion=config.warmup_proportion,
                layerwise_lr_decay_power=config.layerwise_lr_decay,
                n_transformer_layers=model.bert_config.num_hidden_layers,
            )
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[
                    training_utils.ETAHook(
                        {} if config.use_tpu else dict(loss=model.loss),
                        num_train_steps,
                        config.iterations_per_loop,
                        config.use_tpu,
                        10,
                    )
                ],
            )
        else:
            assert mode == tf.estimator.ModeKeys.PREDICT
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=utils.flatten_dict(model.outputs), scaffold_fn=scaffold_fn
            )

        utils.log("Building complete")
        return output_spec

    return model_fn


class ModelRunner(object):
    """Fine-tunes a model on a supervised task."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks, pretraining_config=None):
        self._config = config
        self._tasks = tasks
        self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_cluster_resolver = None
        if config.use_tpu and config.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                config.tpu_name, zone=config.tpu_zone, project=config.gcp_project
            )
        tpu_config = tf.contrib.tpu.TPUConfig(
            iterations_per_loop=config.iterations_per_loop,
            num_shards=config.num_tpu_cores,
            per_host_input_for_training=is_per_host,
            tpu_job_name=config.tpu_job_name,
        )
        # ADDED:限制GPU
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=config.model_dir,
            save_checkpoints_steps=config.save_checkpoints_steps,
            keep_checkpoint_max=5,
            save_checkpoints_secs=None,
            tpu_config=tpu_config,
            session_config=tf_config,
        )

        if self._config.do_train:
            (self._train_input_fn, self.train_steps) = self._preprocessor.prepare_train()
        else:
            self._train_input_fn, self.train_steps = None, 0
        model_fn = model_fn_builder(
            config=config, tasks=self._tasks, num_train_steps=self.train_steps, pretraining_config=pretraining_config
        )
        self._estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            predict_batch_size=config.predict_batch_size,
        )

    def train(self):
        utils.log("Training for {:} steps".format(self.train_steps))
        self._estimator.train(input_fn=self._train_input_fn, max_steps=self.train_steps)

    def evaluate(self, split="dev"):
        return {task.name: self.evaluate_task(task, split=split) for task in self._tasks}

    def evaluate_task(self, task, split="dev", return_results=True):
        """Evaluate the current model."""
        utils.log("Evaluating", task.name, split)
        eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
        results = self._estimator.predict(
            input_fn=eval_input_fn, yield_single_examples=False
        )  # EDITED: edit from True to False
        if task.name == "cmrc2018" or task.name == "drcd":
            scorer = task.get_scorer(split)
        else:
            scorer = task.get_scorer()
        for r in results:
            # pdb.set_trace()
            if r["task_id"] != len(self._tasks):  # ignore padding examples
                r = utils.nest_dict(r, self._config.task_names)
                scorer.update(r[task.name])
        if return_results:
            utils.log(task.name + ": " + scorer.results_str())
            utils.log()
            return dict(scorer.get_results())
        else:
            return scorer

    def write_classification_outputs(self, tasks, trial, split):
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        label2id = {}
        # pdb.set_trace()
        with open(self._config.label_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                label2id[line.strip().split("\t")[0]] = i
        id2label = {v: k for k, v in label2id.items()}
        predict_examples = tasks[0].get_examples(split)
        # print(len(predict_examples))
        statistic_obj = StatisticHelper("test")
        statistic_tags = {}
        """Write classification predictions to disk."""
        utils.log("Writing out predictions for", tasks, split)
        predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
        # pdb.set_trace()
        print("pre_examples", predict_input_fn)
        results_all = self._estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
        print("result", results_all)
        # print("result:",result)

        #    print(len(result))
        if self._config.do_train:
            output_predict_file = os.path.join(self._config.raw, "testResult.transform")
        else:
            output_predict_file = os.path.join(self._config.result_dir, "predict.txt")
        print("output_predict_file:", output_predict_file)
        print("self._config.do_train:", self._config.do_train)
        # predict_lines = processor._read_tsv(FLAGS.data_dir)

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            if self._config.do_train:
                output_lines = []
            else:
                output_lines = ["query\tlabel\n"]
            tf.logging.info("***** Predict results *****")
            if tasks[0].name == "singlelabel":
                for (i, prediction) in enumerate(results_all):
                    # print("prediction:",prediction)
                    if i % 1000 == 0:
                        print("i = %d", i)
                    # pdb.set_trace()
                    # print(i)
                    if i >= len(predict_examples):
                        break
                    predict_example = predict_examples[i]
                    # pdb.set_trace()
                    predict_id = prediction["singlelabel_predictions"]
                    # print("============")
                    # print(id2label)
                    # print(predict_id)
                    predict_label = id2label[int(predict_id)]
                    soft_p = softmax(prediction["singlelabel_logits"])
                    p = sorted(soft_p, reverse=True)
                    label_top_3_to_show = [id2label[np.where(soft_p == i)[0][0]] for i in p[:3]]
                    # label_top_3_to_show = [id2label[soft_p.index(i)] for i in p[:3]]
                    if self._config.do_train:
                        output_line = (
                            str(predict_example.eid)
                            + "\t"
                            + str(predict_example.text_a)
                            + "\t"
                            + str(predict_example.label)
                            + "\t"
                            + str(predict_label)
                            + "\t"
                            + "\t".join(str(class_probability) for class_probability in p[:3])
                            + "\t"
                            + "\t".join(top_label for top_label in label_top_3_to_show)
                            + "\n"
                        )
                    else:
                        output_line = str(predict_example.text_b) + "\t" + str(predict_label) + "\n"
                    output_lines.append(output_line)
                    num_written_lines += 1
                    statistic_obj.cntRightSen() if predict_label == predict_example.label else statistic_obj.cntWrongSen()
                    if predict_label not in statistic_tags:
                        statistic_tags[predict_label] = TagStatisticHelper(predict_label)
                    statistic_tag_pre = statistic_tags[predict_label]
                    if predict_example.label not in statistic_tags:
                        statistic_tags[predict_example.label] = TagStatisticHelper(predict_example.label)
                    statistic_tag = statistic_tags[predict_example.label]

                    if predict_label == predict_example.label:
                        statistic_tag.cntTPSen()
                    else:
                        statistic_tag.cntFNSen()
                        statistic_tag_pre.cntFPSen()
                utils.log("Writing out predictions to", output_predict_file)
                for output_line in output_lines:
                    writer.write(output_line)
                for key, statistic_tag in statistic_tags.items():
                    print("the rest dataset acc are:%s" % statistic_tag.toString())
                if self._config.do_train:
                    test_acc_file = os.path.join(self._config.raw, "test_acc")
                else:
                    test_acc_file = os.path.join(self._config.result_dir, "test_acc")
                with open(test_acc_file, "a", encoding="utf-8") as f:
                    f.write("%.4f\n" % (statistic_obj.getAcc()))
                    f.write("testId\t%s\n" % (self._config.test_id))
                    f.write("time\t%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    for tag, tagStatistic in statistic_tags.items():
                        precision, recall = tagStatistic.getPrecisionAndRecall()
                        f.write(
                            "%s\t%.4f\t%d\t%d\t%.4f\t%d\t%d\n"
                            % (
                                tag,
                                precision,
                                tagStatistic._TP,
                                tagStatistic._TP + tagStatistic._FP,
                                recall,
                                tagStatistic._TP,
                                tagStatistic._TP + tagStatistic._FN,
                            )
                        )
            else:
                # result_clone = copy(results_all)
                scorer = tasks[0].get_scorer()
                # for r in result_clone:
                #     # pdb.set_trace()
                #     if r["task_id"] != len(self._tasks):  # ignore padding examples
                #         r = utils.nest_dict(r, self._config.task_names)
                #         scorer.update(r[tasks[0].name])

                # utils.log(tasks[0].name + ": " + scorer.get_results_outputall())

                for (i, prediction) in enumerate(results_all):
                    if prediction["task_id"] != len(self._tasks):  # ignore padding examples
                        r = utils.nest_dict(prediction, self._config.task_names)
                        scorer.update(r[tasks[0].name])
                    # print("prediction:",prediction)
                    if i % 1000 == 0:
                        print("i = %d", i)
                    # pdb.set_trace()
                    # print(i)
                    predict_example = predict_examples[i]
                    predict_id = prediction["multilabel_predictions"]
                    predict_label = np.where(predict_id == 1)[1].tolist()
                    true_lst = predict_examples[i].label_lst
                    p = sorted(softmax(prediction["multilabel_logits"][0]), reverse=True)
                    # pdb.set_trace()
                    pred_lst = []
                    for i in predict_label:
                        pred_lst.append(id2label[i])
                    true_lst.sort()
                    pred_lst.sort()
                    true = "/".join(true_lst)
                    # pdb.set_trace()
                    pred = "/".join(pred_lst)
                    # pdb.set_trace()
                    output_line1 = (
                        str(predict_example.eid)
                        + "\t"
                        + str(predict_example.text_a)
                        + "\t"
                        + true
                        + "\t"
                        + pred
                        + "\t"
                        + "\t".join(str(class_probability) for class_probability in p)
                        + "\n"
                    )
                    # print(output_line1)
                    output_lines.append(output_line1)

                utils.log("Writing out predictions to", output_predict_file)
                result_mutiple = scorer.get_results_outputall()
                scorer.results_str
                for output_line in output_lines:
                    writer.write(output_line)
                test_acc_file = os.path.join(self._config.raw, "test_acc")
                # pdb.set_trace()
                with open(test_acc_file, "a") as f:
                    f.write("%.4f\n" % (result_mutiple.get("accur")))
                    f.write("testId\t%s\n" % (self._config.test_id))
                    f.write("time\t%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    os.chdir(self._config.raw_data_dir(tasks[0].name))
                    # with open("label.tsv", "r") as d:
                    #     lst = d.readlines()
                    # for tag in lst:
                    precision, recall = result_mutiple.get("precision"), result_mutiple.get("recall")
                    f.write(
                        "multilabel"
                        + "\t"
                        + "%.4f\t%.4f\t%d\n" % (precision, recall, result_mutiple.get("TP") + result_mutiple.get("FN"))
                    )


def write_results(config: configure_finetuning.FinetuningConfig, results):
    """Write evaluation metrics to disk."""
    utils.log("Writing results to", config.results_txt)
    utils.mkdir(config.results_txt.rsplit("/", 1)[0])
    utils.write_pickle(results, config.results_pkl)
    with tf.gfile.FastGFile(config.results_txt, "w") as f:
        results_str = ""
        for trial_results in results:
            for task_name, task_results in trial_results.items():
                if task_name == "time" or task_name == "global_step":
                    continue
                results_str += (
                    task_name + ": " + " - ".join(["{:}: {:.2f}".format(k, v) for k, v in task_results.items()]) + "\n"
                )
        f.write(results_str)
    utils.write_pickle(results, config.results_pkl)


def run_finetuning(config: configure_finetuning.FinetuningConfig):
    """Run finetuning."""
    # Setup for training
    results = []
    tf.ConfigProto().gpu_options.allow_growth = True
    trial = 1
    heading_info = "model={:}, trial {:}/{:}".format(config.model_name, trial, config.num_trials)
    heading = lambda msg: utils.heading(msg + ": " + heading_info)
    heading("Config")
    utils.log_config(config)
    generic_model_dir = config.model_dir
    tasks = task_builder.get_tasks(config)

    # Train and evaluate num_trials models with different random seeds
    while config.num_trials < 0 or trial <= config.num_trials:
        config.model_dir = generic_model_dir
        # if config.do_train:
        #   utils.rmkdir(config.model_dir)

        model_runner = ModelRunner(config, tasks)
        if config.do_train:
            heading_info = "model={:}, trial {:}/{:}".format(config.model_name, trial, config.num_trials)
            heading("Start training")
            model_runner.train()
            utils.log()
        if config.do_eval:
            if config.write_test_outputs and trial <= config.n_writes_test:
                heading("Running on the dev set and writing the predictions")
                for task in tasks:
                    # Currently only writing preds for GLUE and SQuAD 2.0 is supported
                    if task.name in [
                        "cola",
                        "mrpc",
                        "mnli",
                        "sst",
                        "rte",
                        "qnli",
                        "qqp",
                        "sts",
                        "multilabel",
                        "singlelabel",
                    ]:
                        # for split in task.get_test_splits():
                        #   print("split_eval:",split)
                        model_runner.write_classification_outputs([task], trial, "dev")
                    else:
                        utils.log("Skipping1 task", task.name, "- writing predictions is not supported for this task")
            else:
                heading("Run dev set evaluation")
                results.append(model_runner.evaluate(split="dev"))
                # write_results(config, results)

        if config.do_test:
            if config.write_test_outputs and trial <= config.n_writes_test:
                heading("Running on the test set and writing the predictions")
                for task in tasks:
                    print(task)
                    # Currently only writing preds for GLUE and SQuAD 2.0 is supported
                    if task.name in [
                        "cola",
                        "mrpc",
                        "mnli",
                        "sst",
                        "rte",
                        "qnli",
                        "qqp",
                        "sts",
                        "multilabel",
                        "singlelabel",
                    ]:
                        # for split in task.get_test_splits():
                        #   print("split_test:",split )
                        model_runner.write_classification_outputs([task], trial, "test")
                    else:
                        utils.log("Skipping task", task.name, "- writing predictions is not supported for this task")
            else:
                heading("Run test set evaluation")
                results.append(model_runner.evaluate(split="eval"))
                # write_results(config, results)

        if trial != config.num_trials and (not config.keep_all_models):
            utils.rmrf(config.model_dir)
        trial += 1


def main():
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--data_dir", required=True,
    #                     help="Location of data files (model weights, etc).")
    # parser.add_argument("--model_name", required=True,
    #                     help="The name of the model being fine-tuned.")
    # parser.add_argument("--hparams", default="{}",
    #                     help="JSON dict of model hyperparameters.")
    # args = parser.parse_args()
    # if args.hparams.endswith(".json"):
    #   hparams = utils.load_json(args.hparams)
    # else:
    #   hparams = json.loads(args.hparams)
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # run_finetuning(configure_finetuning.FinetuningConfig(
    #     args.model_name, args.data_dir, **hparams))

    # def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_id", default="5005")
    parser.add_argument("--test_id", default="5004")
    parser.add_argument("--data_dir", default="./", help="Location of data files (model weights, etc).")
    parser.add_argument("--model_name", default="small", help="The name of the model being fine-tuned.")
    parser.add_argument("--feature_col", default=1)
    parser.add_argument("--tag_col", default=3)
    parser.add_argument("--raw_data_dir", default="./", help="Location of data files (model weights, etc).")
    parser.add_argument("--hparams", default="{}", help="JSON dict of model hyperparameters.")
    parser.add_argument("--model_dir", default="./", help="Location of data files (model weights, etc).")
    parser.add_argument("--result_dir", default="./", help="Location of data files (model result, etc).")
    parser.add_argument("--pretrain_dir", default="./", help="Pretrain model dir")

    args = parser.parse_args()
    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams.replace("'", '"'))

    tf.logging.set_verbosity(tf.logging.ERROR)
    run_finetuning(
        configure_finetuning.FinetuningConfig(
            args.train_id,
            args.test_id,
            args.data_dir,
            args.model_name,
            args.feature_col,
            args.tag_col,
            args.raw_data_dir,
            args.model_dir,
            args.result_dir,
            args.pretrain_dir,
            **hparams
        )
    )


if __name__ == "__main__":
    main()
