#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:
#
# Description:
# Version:       1.0
# Company:       www.10jqka.com.cn
#
# -------------------------------------------#
import json
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
import os


class EvalImpl:
    def do_eval(
        self,
        predict_file_dir: Path,
        groundtruth_file_dir: Path,
        result_json_file: Path,
        result_detail_file: Path,
        *args,
        **kargs
    ):
        """评测主函数

        Args:
            predict_file_dir (Path): input, 模型预测结果的文件目录
            groundtruth_file_dir (Path): input, 真实结果的文件目录
            result_json_file (Path): output, 评测结果，json格式，{"f1": 0.99}
            result_detail_file (Path): output, 预测明细，可选
        """
        print("Eval begin!!")

        pred_map = {}
        pred_data = pd.read_csv(os.path.join(predict_file_dir, "predict.txt"), sep="\t")
        pred_data = pred_data.reset_index(drop=True)
        print(pred_data.shape)
        for i in range(len(pred_data)):
            query = str(pred_data.loc[i, "query"]) + "_" + str(i)
            label = str(pred_data.loc[i, "label"])
            if query not in pred_map.keys():
                pred_map[query] = label

        test_data = pd.read_csv(os.path.join(groundtruth_file_dir, "test.txt"), sep="\t")
        pred_list = []
        test_list = list(test_data["label"])
        for i in range(len(test_data)):
            query = str(test_data.loc[i, "query"]) + "_" + str(i)
            if query in pred_map.keys():
                pred_list.append(pred_map[query])
            else:
                pred_list.append("")

        # label_list = sorted(list(set(test_list)))
        label_list = ["未攻击用户", "攻击用户"]
        label_map = dict()
        for i in range(len(label_list)):
            label_map[label_list[i]] = i

        test_list = [label_map.get(one, 0) for one in test_list]
        pred_list = [label_map.get(one, 0) for one in pred_list]
        f1 = f1_score(test_list, pred_list)

        result = {"f1": f1}
        print(result)
        with open(result_json_file, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(result))

        pred_data.to_csv(result_detail_file, header=True, index=False, encoding="utf8", sep="\t")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_dir", type=str, default="../../groundtruth/")
    parser.add_argument("--groundtruth_file_dir", type=str, default="../../groundtruth/")
    parser.add_argument("--result_json_file", type=str, default="../../output/")
    parser.add_argument("--result_detail_file", type=str, default="")

    parser.add_argument("--model_name", default="sac", help="The name of the model being fine-tuned.")
    args = parser.parse_args()
    eval_object = EvalImpl()
    eval_object.do_eval(
        args.predict_file_dir, args.groundtruth_file_dir, args.result_json_file, args.result_detail_file
    )
