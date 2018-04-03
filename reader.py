#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/1/9下午3:39
# * file name   :  reader.py


import jieba.posseg
import re
import os
import codecs
import string
from zhon.hanzi import punctuation
import jieba


htmltag = ['症状和体征', '检查和检验', '治疗', '疾病和诊断', '身体部位']
englishtag = ['SYMPTOM', 'CHECK', 'TREATMENT', 'DISEASE', 'BODY']


def readFileUTF8(filename):
    fr = codecs.open(filename, 'r', 'utf-8')
    text = ''
    for line in fr:
        text += line.strip()
    return text


def readData(filename):
    datas = []
    data = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                data.append(fields)
            else:
                datas.append(data)
                data = []
    if len(data) != 0:
        datas.append(data)
    return datas


def extract_tag_information(text):
    res = {}
    for i, html in enumerate(htmltag):
        res[englishtag[i]] = []
        pattern = re.compile(r'<' + html + '>(.*?)</' + html + '>', re.S)
        contents = pattern.findall(text)
        for content in contents:
            content = re.compile('<[^>]+>', re.S).sub('', content)
            res[englishtag[i]].append(content)
    return res


def extract_all_information(text):
    pattern = re.compile('<(.*?)>(.*?)</\\1>', re.S)
    contents = pattern.findall(text)
    ans = ''
    for content in contents:
        content = re.compile('<[^>]+>', re.S).sub('', content[1])
        ans += content
        print(content)
    return ans


def getType(type):
    if type == '症状和体征':
        return 'SIGNS'
    elif type == '检查和检验':
        return 'CHECK'
    elif type == '疾病和诊断':
        return 'DISEASE'
    elif type == '治疗':
        return 'TREATMENT'
    elif type == '身体部位':
        return 'BODY'
    else:
        return 'OTHER'


def getWord(tag):
    """
    将ner中对应的标签，转化为文字
    :param tag:
    :return:
    """
    if tag == 'body':
        return '身体部位'
    elif tag == 'chec':
        return '疾病和诊断'
    elif tag == 'cure':
        return '治疗'
    elif tag == 'dise':
        return '疾病和诊断'
    elif tag == 'symp':
        return '症状和体征'
    else:
        return 'other'


def split(text):
    """以标签数据分割成list"""
    res = []
    start = 0
    end = 0
    while end < len(text):
        if text[end] == '<':
            # < 前面的信息写入
            if start != end:
                res.append(text[start: end])
                start = end + 1
            else:
                start += 1
            # <>中的信息
            end = go(text, start)
            res.append(text[start: end])
            start = end + 1
            end = start
        else:
            end += 1
    if start != end:
        res.append(text[start: end])
    return res


def go(text, i):
    while i < len(text):
        if text[i] == '>':
            break
        else:
            i += 1
    return i


def text2word_ner_biose_format(text):
    """
    将标签数据集转换成word_ner_BIOSE格式的标准数据集
    :param text:
    :return:
    """
    # 过滤掉所有的标签
    # content = re.compile('<[^>]+>', re.S).sub('', text)
    segment = jieba.posseg.cut(text)
    # 采用BIOSE方式
    # B: 开始，I：中间，O：无关词，S：单个词，E：结尾
    # 将训练数据转换为标准的ner格式的数据
    start = 0
    type = ''
    stack = []
    flag = 0
    features = []
    pieces = split(text)
    pre = 0
    for seg in segment:
        if seg.word == '<':
            flag = 1
            pre = 0
            continue
        elif seg.word == '>':
            flag = 0
            pre = 0
            continue

        if flag == 0:
            while start < len(pieces) and getType(pieces[start]) != 'OTHER':
                stack.append(getType(pieces[start]))
                start += 1
            while start < len(pieces) and getType(pieces[start][1:]) != 'OTHER':
                stack.pop()
                start += 1
            while start < len(pieces) and getType(pieces[start]) != 'OTHER':
                stack.append(getType(pieces[start]))
                start += 1
            index = pieces[start].find(seg.word, pre)
            pre = index + 1
            if len(stack) == 0:
                type = 'O'
                if start < len(pieces) and index + len(seg.word) == len(pieces[start]):
                    start += 1
            else:
                if start < len(pieces):
                    if index == 0 and len(seg.word) == len(pieces[start]):
                        type = 'S-' + stack[-1]
                        start += 1
                    elif index == 0 and len(seg.word) != len(pieces[start]):
                        type = 'B-' + stack[-1]
                    elif index != -1 and len(pieces[start]) - index == len(seg.word):
                        if start + 1 == len(pieces) or getType(pieces[start + 1]) == 'OTHER':
                            type = 'E-' + stack[-1]
                        else:
                            type = 'I-' + stack[-1]
                        start += 1
                    elif index != -1:
                        type = 'I-' + stack[-1]

            features.append([seg.word, seg.flag, type])
    return features


def text2word_ner_bio_format(text):
    """
    将标签数据集转换成word_ner_BIO格式的标准数据集
    :param text:
    :return:
    """
    segment = jieba.posseg.cut(text)
    # 采用BIOSE方式
    # B: 开始，I：中间，O：无关词，S：单个词，E：结尾
    # 将训练数据转换为标准的ner格式的数据
    start = 0
    type = ''
    stack = []
    flag = 0
    features = []
    pieces = split(text)
    pre = 0
    for seg in segment:
        if seg.word == '<':
            flag = 1
            pre = 0
            continue
        elif seg.word == '>':
            flag = 0
            pre = 0
            continue

        if flag == 0:
            while start < len(pieces) and getType(pieces[start]) != 'OTHER':
                stack.append(getType(pieces[start]))
                start += 1
            while start < len(pieces) and getType(pieces[start][1:]) != 'OTHER':
                stack.pop()
                start += 1
            while start < len(pieces) and getType(pieces[start]) != 'OTHER':
                stack.append(getType(pieces[start]))
                start += 1
            if start < len(pieces):
                index = pieces[start].find(seg.word, pre)
                pre = index + 1
                if len(stack) == 0:
                    type = 'O'
                    if start < len(pieces) and index + len(seg.word) == len(pieces[start]):
                        start += 1
                else:
                    if start < len(pieces):
                        if index == 0:
                            type = 'B-' + stack[-1]
                        elif index != -1:
                            type = 'I-' + stack[-1]
                        if len(pieces[start]) - index == len(seg.word):
                            start += 1
                features.append([seg.word, seg.flag, type])
    return features


def text2char_ner_bio_format(text):
    """
    将数据转化为标注的char_bio_format格式的数据集
    :param text:
    :return:
    """
    segment = list(text)
    stack = []
    features = []
    start = 0
    end = len(segment)
    label = ''
    first = 0  # first表示当前的字符是否是第一个字符
    while start < end:
        if segment[start] == '<':
            tag = ''
            start += 1
            if segment[start] != '/':
                while start < end and segment[start] != '>':
                    tag += segment[start]
                    start += 1
                stack.append(getType(tag))
            else:
                start += 1
                while start < end and segment[start] != '>':
                    start += 1
                    tag += segment[start]
                stack.pop()
            first = 1
        else:
            if len(stack) == 0:
                label = 'O'
            else:
                if first == 1:
                    label = 'B-' + stack[-1]
                    first = 0
                else:
                    label = 'I-' + stack[-1]
            features.append([segment[start], label])
        start += 1
    return features


# 将标注过的ner数据集，提取出实体
def getNamedEntity(word, ner):
    ans = []
    cur = ''
    for i, tag in enumerate(ner):
        if 'B' == tag.split('-')[0]:
            cur += word[i]
        elif 'I' == tag.split('-')[0]:
            cur += word[i]
        elif 'E' == tag.split('-')[0]:
            cur += word[i]
            ans.append(cur)
            cur = ''
        elif 'S' == tag.split('-')[0]:
            if len(cur) == 0:
                ans.append(word[i])
            else:
                cur += word[i]
    return ans


def charbio2text(data):
    """
    将ner三元组形式转换成标签的形式
    :param data:
    :return:
    """
    content = ''
    current = ''
    for word, pos, label in data:
        fields = label.split('-')
        if len(fields) == 2:
            position, type = fields
            if position == 'B':
                if len(current) != 0:
                    content += '</' + current + '>'
                current = getWord(type)
                content += '<' + current + '>' + word
            elif position == 'I':
                content += word
        else:
            if len(current) != 0:
                content += '</' + current + '>'
                current = ''
            content += word
    return content


if __name__ == '__main__':
    # fw = open('../train_test_data/train_bio_word.txt', 'w', encoding='utf-8')
    # count = 0
    # for i in range(1, 81):
    #     filename = '../raw_data/病史特点-' + str(i) + '.txt'
    #     answer = text2word_ner_bio_format(readFileUTF8(filename))
    #     for [word, pos, ner] in answer:
    #         fw.write(word + '\t' + pos + '\t' + ner + '\n')
    #         count += 1
    #         if count == 60:
    #             fw.write('\n')
    #             count = 0
    #     fw.write('\n')
    #     print('file ' + str(i) + ' has already finished!')
    # fw.flush()
    # fw.close()

    # text = readFileUTF8('../raw_data/病史特点-80.txt')
    # contents = text2char_ner_bio_format(text)
    # words = [word for word, tag in contents]
    # tags = [tag for word, tag in contents]
    # entity = crf.predata.get_entity(tags, words, ['SIGNS', 'CHECK', 'DISEASE', 'TREATMENT', 'BODY'])
    # print(entity)

    # datas = readData('/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/char_data/test_BIO.txt')
    # print(len(datas))
    # with open('/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/label_data/test_label.txt', 'w',
    #           encoding='utf-8') as fw:
    #     for data in datas:
    #             fw.write(charbio2text(data) + '\n')

    # parent = '/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/label_data'
    # fw = open(os.path.join(parent, 'test_word.txt'), 'w', encoding='utf-8')
    # with open(os.path.join(parent, 'test_label.txt'), 'r', encoding='utf-8') as fr:
    #     for line in fr:
    #         features = text2word_ner_bio_format(line.strip())
    #         for word, pos, tag in features:
    #             fw.write(word + '\t' + pos + '\t' + tag + '\n')
    #         fw.write('\n')
    # fw.close()

    data_all = open('/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/data_char_all.txt',
                    'w', encoding='utf-8')
    raw_datas = []
    with open('/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/train_test_data/train_bio_word.txt',
              'r', encoding='utf-8') as fr:
        data = ''
        for line in fr:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                data += fields[0]
            elif len(data) != 0:
                raw_datas.append(data)
                data = ''
    if len(data) != 0:
        raw_datas.append(data)
        data = ''
    with open('/Users/baiyyang/PycharmProjects/python3/medical_entity_recognition/train_test_data/test_bio_word.txt',
              'r', encoding='utf-8') as fr:
        for line in fr:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                data += fields[0]
            elif len(data) != 0:
                raw_datas.append(data)
                data = ''
    if len(data) != 0:
        raw_datas.append(data)
        data = ''
    for data in raw_datas:
        # segments = jieba.cut(data, cut_all=False)
        segments = list(data)
        words = [seg for seg in segments if seg not in punctuation and seg not in string.punctuation]
        if len(words) < 2:
            continue
        data_all.write(' '.join(words) + '\n')
    data_all.close()



