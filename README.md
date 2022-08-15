# Chinese-financial-NER（金融领域命名实体识别）



# Project address：（项目代码地址）

由于预训练模型过大，我们将整个项目代码放在百度云盘里供用户下载

Because the project file is too large, we put the project in Baidu cloud disk. Thank you for your understanding

链接：https://pan.baidu.com/s/1lUn6EC-gD5-E6g8EkAklwQ 
提取码：M978 








# How to run this project（如何运行我们的项目）

run simple.sh
 
 
 
 
 
python mcasp_main.py --do_train --train_data_path=./sample_data/train.txt --dev_data_path=./sample_data/dev.txt --test_data_path=./sample_data/test.txt --use_zen --bert_model=./ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 30 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=models/test_zen
