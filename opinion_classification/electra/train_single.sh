#export CUDA_VISIBLE_DEVICES="1";python3 run_finetuning.py --data_dir data --model_name small --hparams '{"model_size":"small","task_names":["singlelabel"],"do_train":true,"do_eval":true,"do_test":true}'
#export CUDA_VISIBLE_DEVICES="0" ;python3 run_finetuning.py  --train_id 3551 --test_id  3552 --data_dir /data2/classifier/bootdo/models/ELECTRA_master/data --model_name small --feature_col 1 --tag_col 3 --raw_data_dir /data2/classifier/bootdo/data/train//train.942.3551.3552/ --hparams '{"continue_train":false,"do_train":true,"do_eval":true,"do_test":true,"task_names":["singlelabel"],"model_size":"small"}'
python3 run_finetuning.py  --train_id 5005 --test_id  5004 --data_dir /data2/classifier/bootdo/models/electra_master_en/data --model_name small --feature_col 1 --tag_col 3 --raw_data_dir /data2/classifier/bootdo/data/train/train.1007.5005.5004/result_ELECTRA --hparams '{"continue_train":false,"do_train":true,"do_eval":true,"do_test":true,"task_names":["singlelabel"],"model_size":"small"}'