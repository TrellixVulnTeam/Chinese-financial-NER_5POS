mkdir logs
mkdir models

MODEL_HOME=/path/to/pretrained_models


#python mcasp_main.py --do_train --train_data_path=./sample_data/train.tsv --dev_data_path=./sample_data/dev.tsv --test_data_path=./sample_data/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=2 --num_train_epochs 3 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=models/test_bert



python mcasp_main.py --do_train --train_data_path=./sample_data/train.txt --dev_data_path=./sample_data/dev.txt --test_data_path=./sample_data/test.txt --use_zen --bert_model=./ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 30 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=models/test_zen






