tf1.11python run_save_pb_classifier.py -bert_model_dir not_in_use -model_dir data/models/small/finetuning_models/singlelabel_model_1  -model_pb_dir ./pb_dir/single -max_seq_len 40 -num_labels 3 --data_dir data --model_name small --hparams '{"model_size":"small","task_names":["singlelabel"]}'
