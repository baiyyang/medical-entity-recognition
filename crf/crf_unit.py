#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/1/10上午10:29
# * file name   :  crf_unit.py


import codecs
import pycrfsuite
import string
import zhon.hanzi as zh
import crf.predata
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


# 获取数据
def readData(filename):
    fr = codecs.open(filename, 'r', 'utf-8')
    data = []
    for line in fr:
        fields = line.strip().split('\t')
        if len(fields) == 3:
            data.append(fields)
    return data


testpath = ''
trainpath = ''


test = readData(testpath)
train = readData(trainpath)


# 判断是否为标点符号
# punctuation
def ispunctuation(word):
    punctuation = string.punctuation + zh.punctuation
    if punctuation.find(word) != -1:
        return True
    else:
        return False


# 特征定义
def word2features(sent, i):
    """返回特征列表"""
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word=' + word,
        'word_tag=' + postag,
    ]
    if i > 0:
        features.append('word[-1]=' + sent[i-1][0])
        features.append('word[-1]_tag=' + sent[i-1][1])
        if i > 1:
            features.append('word[-2]=' + sent[i-2][0])
            features.append('word[-2, -1]=' + sent[i-2][0] + sent[i-1][0])
            features.append('word[-2]_tag=' + sent[i-2][1])
    if i < len(sent) - 1:
        features.append('word[1]=' + sent[i+1][0])
        features.append('word[1]_tag=' + sent[i+1][1])
        if i < len(sent) - 2:
            features.append('word[2]=' + sent[i+2][0])
            features.append('word[1, 2]=' + sent[i+1][0] + sent[i+2][0])
            features.append('word[2]_tag=' + sent[i+2][1])
    return features


def sent2feature(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2label(sent):
    return [label for word, pos, label in sent]


def sent2word(sent):
    return [word for word, pos, label in sent]


X_train = sent2feature(train)
y_train = sent2label(train)

X_test = sent2feature(test)
y_test = sent2label(test)

# 训练模型
model = pycrfsuite.Trainer(verbose=True)
model.append(X_train, y_train)
model.set_params({
    'c1': 1.0,  # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True,
    'feature.minfreq': 3
})

model.train('./medical_entity_recognition_bio_word_ori.crfsuite')


# 预测数据
tagger = pycrfsuite.Tagger()
tagger.open('./medical_entity_recognition_bio_word_ori.crfsuite')


# 评估模型
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a l ist of BIOSE-encoded sequences.
    It computes token-level metrics and discards 'O' labels.
    :param y_true:
    :param y_pred:
    :return:
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(y_true)
    y_pred_combined = lb.transform(y_pred)

    tagset = set(lb.classes_) - {'O'}
    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {
        cls: idx for idx, cls in enumerate(lb.classes_)
    }

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset
    )


y_pred = list(tagger.tag(X_test))
entity = ['BODY', 'CHECK', 'DISEASE', 'SIGNS', 'TREATMENT']
entities_pred = crf.predata.get_entity(y_pred, sent2word(test), entity)
entities_true = crf.predata.get_entity(y_test, sent2word(test), entity)
pre_all = 0
rec_all = 0
f1_all = 0
for i, (pred, true) in enumerate(zip(entities_pred, entities_true)):
    pre = 0
    rec = 0
    for p in pred:
        if p in true:
            pre += 1
    for t in true:
        if t in pred:
            rec += 1
    precision = pre * 1.0 / len(pred)
    recall = rec * 1.0 / len(true)
    f1 = 2 * precision * recall / (precision + recall)
    pre_all += precision
    rec_all += recall
    f1_all += f1
    print('{:10s}: precision:{:.2f}, recall:{:.2f}, f1-score:{:.2f}'.format(entity[i], precision, recall, f1))

print('{:10s}: precision:{:.2f}, recall:{:.2f}, f1-score:{:.2f}'.format(
    'average', pre_all / len(entity), rec_all / len(entity), f1_all / len(entity)))

# print(bio_classification_report(y_test, y_pred))


# 测试提取的结果
# content = readData(testpath)
# ans = list(tagger.tag(sent2feature(content)))
# print(sent2word(content))
# print(sent2label(content))
# print(ans)
# pre_informations = crf.predata.get_entity(ans, sent2word(content), ['body', 'chec', 'cure', 'dise', 'symp'])
# test_informations = crf.predata.get_entity(sent2label(content), sent2word(content),
#                                            ['body', 'chec', 'cure', 'dise', 'symp'])
# body, chec, cure, dise, symp = pre_informations
# print('body:{}\nchec:{}\ncure:{}\ndise:{}\nsymp:{}\n'.format(body, chec, cure, dise, symp))
# print('-' * 20)
# body, chec, cure, dise, symp = test_informations
# print('body:{}\nchec:{}\ncure:{}\ndise:{}\nsymp:{}\n'.format(body, chec, cure, dise, symp))

# content = readData(devpath)
# ans = list(tagger.tag(sent2feature(content)))
# pre_informations = crf.predata.get_entity(ans, sent2word(content), ['LOC', 'ORG', 'PER'])
# test_informations = crf.predata.get_entity(sent2label(content), sent2word(content), ['LOC', 'ORG', 'PER'])
# LOC, ORG, PER = pre_informations
# print('LOC:{}\nORG:{}\nPER:{}\n'.format(LOC, ORG, PER))
# print('-' * 40)
# LOC, ORG, PER = test_informations
# print('LOC:{}\nORG:{}\nPER:{}\n'.format(LOC, ORG, PER))

