#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  将医疗数据集转化为B，I，O三种格式的标准数据
# * create time :  2018/3/8上午10:21
# * file name   :  predata.py

import codecs
import re


def readfile(filename):
    content = ''
    fr = codecs.open(filename, 'r', 'utf-8')
    for line in fr:
        content += line.strip()
    return content


def content2uniformcontent(content):
    length = len(content)
    seq, label = [], []
    i = 0
    tag = 'O'  # label
    pre = 0  # 记录前面是否还有<>开头的信息
    flag = 0  # 记录有两个连续的<>
    while i < length:
        if content[i] != '<':
            seq.append(content[i])
            label.append(tag)
            if tag == 'B':
                tag = 'I'
                flag = 1
            i += 1
        else:
            if content[i + 1] == '/':
                pre -= 1
                if pre == 0:
                    tag = 'O'
                else:
                    tag = 'I'
            elif content[i + 1] != '/':
                pre += 1
                if pre == 1:
                    tag = 'B'
                    flag = 0
                elif flag == 1:
                    tag = 'I'
            while i < length and content[i] != '>':
                i += 1
            i += 1
    return seq, label


# 根据seq和label将对应的信息提取出来
def extract_information(seq, label):
    information = []
    info = ''
    for i, tag in enumerate(label):
        if tag == 'B' or tag == 'I':
            info += seq[i]
        elif tag == 'O' and len(info) != 0:
            information.append(info)
            info = ''
    if len(info) > 0:
        information.append(info)
    return information


# 将标签过滤掉，获取原文本
def extract_raw(content):
    pattern = re.compile('<[^>]+>')
    return pattern.sub('', content)


# 将原始的BIEO类型的标签，改为BIO
def change_tag(origin_path, target_path):
    fw = open(target_path, 'w', encoding='utf-8')
    with open(origin_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                char, pos, tag = fields
                if tag.find('E') != -1:
                    tag = 'I-' + tag.split('-')[1]
                fw.write(char + '\t' + pos + '\t' + tag + '\n')
            else:
                fw.write('\n')
    fw.close()


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
    # content = readfile('./raw_data/病史特点-8.txt')
    # seq, label = content2uniformcontent(content)
    # for i, s in enumerate(seq):
    #     print(s, label[i])
    # information = extract_information(seq, label)
    # print('\n'.join(information))

    # fw = codecs.open('./data_path/test.txt', 'w', 'utf-8')
    # for i in range(81, 101):
    #     content = readfile('./raw_data/病史特点-{}.txt'.format(i))
    #     seq, label = content2uniformcontent(content)
    #     for i, s in enumerate(seq):
    #         if s == '' or label[i] == '':
    #             continue
    #         else:
    #             fw.write(s + '\t' + label[i] + '\n')
    #     fw.write('\n')
    # fw.close()
    # print('finished')

    # content = readfile('./raw_data/病史特点-89.txt')
    # print(extract_raw(content))
    change_tag('data_path/test_new.txt', 'data_path/test_BIO.txt')





