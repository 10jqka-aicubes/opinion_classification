python3 run_save_pb_classifier_xh.py -bert_model_dir not_in_use -model_dir /data2/classifier/bootdo/data/train/train.1015.5090.4970/result_ELECTRA/finetuning_models/multilabel_model_1  -model_pb_dir ./pb_dir/multi -max_seq_len 40  -num_labels 20 --data_dir /data2/classifier/bootdo/models/electra_master_en/data --model_name small --hparams '{"model_size":"small","task_names":["multilabel"]}' --train_id 5090 --test_id  4970 --raw_data_dir /data2/classifier/bootdo/data/train/train.1015.5090.4970/result_ELECTRA --feature_col 1 --tag_col 3
