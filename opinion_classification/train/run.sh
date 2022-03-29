#!/usr/bin/bash
basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath
# 以下是样例，你可以自定义修改
model_data_dir="$SAVE_MODEL_DIR/data/models/data/"
echo "data_dir: $model_data_dir"
python ../electra/run_finetuning.py \
    --raw_data_dir=$TRAIN_FILE_DIR \
    --data_dir=$model_data_dir \
    --model_dir=$SAVE_MODEL_DIR \
    --result_dir=$SAVE_MODEL_DIR \
    --hparams '{"continue_train":false,"do_train":true,"do_eval":false,"do_test":false,"task_names":["singlelabel"],"model_size":"small"}'
