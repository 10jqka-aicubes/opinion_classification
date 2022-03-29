#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

model_data_dir="$SAVE_MODEL_DIR/data/models/data/"
echo "data_dir: $model_data_dir"
# 以下是样例，你可以自定义修改
python ../electra/run_finetuning.py \
    --raw_data_dir=$PREDICT_FILE_DIR \
    --data_dir=$model_data_dir\
    --model_dir=$SAVE_MODEL_DIR \
    --result_dir=$PREDICT_RESULT_FILE_DIR  \
    --hparams '{"continue_train":false,"do_train":false,"do_eval":false,"do_test":true,"task_names":["singlelabel"],"model_size":"small"}'