#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  将数据处理成标准的BIO的形式
# * create time :  2018/3/8下午3:41
# * file name   :  predata.py

import codecs
import jieba
import jieba.posseg
import re


def readcontent(filename):
    content = ''
    with codecs.open(filename, 'r', 'utf-8') as fr:
        for line in fr:
            content += line.strip()
    return content


def get_content(content):
    pattern = re.compile('<[^>]+>')
    return pattern.sub('', content)


def text2ner(text):
    seq, pos, label = [], [], []
    segment = jieba.posseg.cut(text)
    words, flags = [], []
    for seg in segment:
        words.append(seg.word)
        flags.append(seg.flag)
    i = 0
    tag = 'O'
    pre = 0  # 判断前面<>
    sign = 0  # 记录有多个连续的<>
    while i < len(words):
        if words[i] != '<':
            seq.append(words[i])
            pos.append(flags[i])
            label.append(tag)
            if tag == 'B':
                tag = 'I'
                sign = 1
            i += 1
        else:
            if words[i+1] == '/':
                pre -= 1
                if pre == 0:
                    tag = 'O'
                else:
                    tag = 'I'
            else:
                pre += 1
                if pre == 1:
                    tag = 'B'
                    sign = 0
                elif sign == 1:
                    tag = 'I'
            while i < len(words) and words[i] != '>':
                i += 1
            i += 1
    return seq, pos, label


def extract_information(seq, label):
    information = []
    info = ''
    for i, s in enumerate(seq):
        if label[i] == 'B' or label[i] == 'I':
            info += s
        else:
            if len(info) != 0:
                information.append(info)
                info = ''
    if len(info) != 0:
        information.append(info)
    return information


def get_entity(tag_seq, char_seq, keys):
    """
    返回实体类别
    :param tag_seq:
    :param char_seq:
    :param keys: key为list，表示返回需要的类别名称
    :return:
    """
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entity = []
    for key in keys:
        entity.append(get_entity_key(tag_seq, char_seq, key))
    return entity


def get_entity_key(tag_seq, char_seq, key):
    entities = []
    entity = ''
    for (char, tag) in zip(char_seq, tag_seq):
        if tag == 'B-' + key or tag == 'I-' + key or tag == 'E-' + key:
            entity += char
        else:
            if len(entity) != 0:
                entities.append(entity)
                entity = ''
    if len(entity) != 0:
        entities.append(entity)
    return entities


if __name__ == '__main__':
    # text = readcontent('raw_data/病史特点-80.txt')
    # seq, pos, label = text2ner(text)
    # for i, s in enumerate(seq):
    #     print(s, pos[i], label[i])
    # information = extract_information(seq, label)
    # print('\n'.join(information))

    # fw = codecs.open('test1.txt', 'w', 'utf-8')
    # for i in range(1, 2):
    #     text = readcontent('raw_data/病史特点-{}.txt'.format(i))
    #     seq, pos, label = text2ner(text)
    #     for i, s in enumerate(seq):
    #         fw.write(s + '\t' + pos[i] + '\t' + label[i] + '\n')
    # fw.close()
    # print('finished')
    with open('content.txt', 'w', encoding='utf-8') as fw:
        for i in range(1, 101):
            text = readcontent('../raw_data/病史特点-' + str(i) + '.txt')
            text = get_content(text)
            fw.write(text + '\n')





