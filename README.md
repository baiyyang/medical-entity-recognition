# medical-entity-recognition
## Describe
本项目是针对医疗数据，进行命名实体识别。主要采用的方法：

1. 基于条件随机场(Condition Random Fields, CRF)的命名实体识别.

2. 基于双向长短时记忆神经网络和条件随机场(Bi-LSTM-CRF)的命名实体识别。

## Introduce
1. raw_data是原始数据，来源于[CCKS2017](http://www.ccks2017.com/en/index.php/sharedtask/)任务二中，针对医疗电子病例进行命名实体识别。reader.py文件是对原始数据进行处理，生成标准的NER格式(data, pos, label)的数据。

2. train_test_data是模型的训练和测试的语料，其中word2id.pkl和char2id.pkl是神经网络中需要读入的字典。

3. crf文件夹是使用CRF进行命名实体识别的模型，其中medical_entity_recognition_bio_char_ori.crfsuite和medical_entity_recognition_bio_word_ori.crfsuite分别是训练好的，以字为特征单元和词为特征单元的模型。

4. bilstm_crf文件夹中是基于神经网络的命名实体识别的模型。其中，bio_model下存放的是已经训练好的两个模型。分别是随机初始化embedding的字向量和词向量的模型。其中:
  - 训练新的模型方法：
> python main.py --mode train --data_dir *** --train_data *** --test_data *** --dictionary ***

  - 测试已有模型方法:
> python main.py --mode test --data_dir ../train_test_data --train_data train_bio_char.txt --test_data test_bio_char.txt --dictionary char2id.pkl --demo_model random_char_300

## Requirements
python 3

pycrfsuite：pip install python-crfsuite

zhon：pip install zhon

tensorflow >= 1.4

## Result
分别以字和词为单元进行训练，实验结果如下：

|model|char_unit|word_unit|
|:------:|:-----:|:-----:|
|CRF|0.73|0.74|
|Bi-LSTM_CRF|0.80|0.78|

## Reference
[guillaumegenthial/sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)

## Other
欢迎各位大佬，批评指正
