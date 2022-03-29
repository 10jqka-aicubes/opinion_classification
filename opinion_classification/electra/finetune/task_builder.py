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

"""Returns task instances given the task name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configure_finetuning
from finetune.classification import classification_tasks
from finetune.qa import qa_tasks
from finetune.tagging import tagging_tasks
from model import tokenization
import os
import urllib.request
import random
import pdb


def get_tasks(config: configure_finetuning.FinetuningConfig):
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
    return [get_task(config, task_name, tokenizer) for task_name in config.task_names]


def _process(task_name, config: configure_finetuning.FinetuningConfig):
    # pass
    random.seed(1)
    # os.chdir(config.raw_data_dir(task_name))
    if int(config.train_id) and bool(config.do_train) and not bool(config.continue_train):
        lines = []
        with open(config.raw_data_dir + "/train.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                lines.append(line)

        random.shuffle(lines)
        tags = {}
        doc = {}
        for i, line in enumerate(lines):
            if i == 0:
                continue
            lst = line.strip().split("\t")
            if len(lst) < 2:
                continue
            # tag = lst[1]
            tag = lst[-1]
            tag = tokenization.convert_to_unicode(tag)
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] = 1 + tags[tag]
            if tag not in doc:
                doc[tag] = []
            doc[tag].append(line)
        for key, value in doc.items():
            print(key, len(value))
        del_label = ["", "label"]
        for one_label in del_label:
            if one_label in tags.keys():
                del tags[one_label]
        print("全部标签数据：", tags)
        max_num_tag = ""
        max_num = 0
        min_num_tag = ""
        min_num = len(lines)

        with open(config.label_file, "w", encoding="utf-8") as f:
            if config.task_names[0] == "singlelabel":
                for key, value in tags.items():
                    # if value < 10:
                    #   continue
                    f.write("%s\n" % (key))
                    if value > max_num:
                        max_num = value
                        max_num_tag = key
                    if value < min_num:
                        min_num = value
                        min_num_tag = key
                print("max_num_tag", max_num_tag, "min_num_tag", min_num_tag)
            else:

                label_diff = []
                for key in tags.keys():
                    # if value < 10:
                    #   continue
                    tmp_label = key.replace(" ", ",")
                    tmp_label = tmp_label.replace("/", ",")
                    tmp_label = tmp_label.split(",")
                    label_diff += tmp_label
                for label_x in set(label_diff):
                    f.write("%s\n" % (label_x))


def get_tasks(config: configure_finetuning.FinetuningConfig):
    print("vocab_file:", config.vocab_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
    return [get_task(config, task_name, tokenizer) for task_name in config.task_names]


def get_task(config: configure_finetuning.FinetuningConfig, task_name, tokenizer):
    """Get an instance of a task based on its name."""
    if task_name == "cola":
        return classification_tasks.CoLA(config, tokenizer)
    elif task_name == "mrpc":
        return classification_tasks.MRPC(config, tokenizer)
    elif task_name == "mnli":
        return classification_tasks.MNLI(config, tokenizer)
    elif task_name == "sst":
        return classification_tasks.SST(config, tokenizer)
    elif task_name == "rte":
        return classification_tasks.RTE(config, tokenizer)
    elif task_name == "qnli":
        return classification_tasks.QNLI(config, tokenizer)
    elif task_name == "qqp":
        return classification_tasks.QQP(config, tokenizer)
    elif task_name == "sts":
        return classification_tasks.STS(config, tokenizer)
    elif task_name == "squad":
        return qa_tasks.SQuAD(config, tokenizer)
    elif task_name == "squadv1":
        return qa_tasks.SQuADv1(config, tokenizer)
    elif task_name == "newsqa":
        return qa_tasks.NewsQA(config, tokenizer)
    elif task_name == "naturalqs":
        return qa_tasks.NaturalQuestions(config, tokenizer)
    elif task_name == "triviaqa":
        return qa_tasks.TriviaQA(config, tokenizer)
    elif task_name == "searchqa":
        return qa_tasks.SearchQA(config, tokenizer)
    elif task_name == "chunk":
        return tagging_tasks.Chunking(config, tokenizer)
    elif task_name == "bqcorpus":
        return classification_tasks.BQCorpus(config, tokenizer)
    elif task_name == "chnsenticorp":
        return classification_tasks.ChnSentiCorp(config, tokenizer)
    elif task_name == "xnli":
        return classification_tasks.XNLI(config, tokenizer)
    elif task_name == "lcqmc":
        return classification_tasks.LCQMC(config, tokenizer)
    elif task_name == "cmrc2018":
        return qa_tasks.CMRC2018(config, tokenizer)
    elif task_name == "drcd":
        return qa_tasks.DRCD(config, tokenizer)
    elif task_name == "singlelabel":
        _process(task_name, config)
        return classification_tasks.SingleLabel(config, tokenizer)
    elif task_name == "multilabel":
        _process(task_name, config)
        return classification_tasks.MultiLabel(config, tokenizer)
    else:
        raise ValueError("Unknown task " + task_name)
