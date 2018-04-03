import sys
import pickle
import os
import random
import numpy as np

# tags, BIO
# tag2label = {"O": 0,
#              "B": 1,
#              "I": 2
#              }

# tags, BIEO-body, chec, cure, dise, symp
# tag2label = {"B-body": 0, "B-chec": 1, "B-cure": 2, "B-dise": 3, "B-symp": 4,
#              "E-body": 5, "E-chec": 6, "E-cure": 7, "E-dise": 8, "E-symp": 9,
#              "I-body": 10, "I-chec": 11, "I-cure": 12, "I-dise": 13, "I-symp": 14,
#              "O": 15}

# tags, BIO-body, chec, cure, dise, symp
tag2label = {"B-BODY": 0, "B-CHECK": 1, "B-SIGNS": 2, "B-DISEASE": 3, "B-TREATMENT": 4,
             "I-BODY": 5, "I-CHECK": 6, "I-SIGNS": 7, "I-DISEASE": 8, "I-TREATMENT": 9,
             "O": 10}


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: raw_data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            if len(line.strip().split('\t')) != 0:
                fields = line.strip().split('\t')
                char = fields[0]
                label = fields[-1]
                sent_.append(char)
                tag_.append(label)
        elif len(sent_) != 0 and len(tag_) != 0:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    # 删除词频出现较少的单词
    for word in low_freq_words:
        del word2id[word]

    # 剔除词频出现较少的单词，并且重新编号
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    # 将word2id存储进入文件中
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    将一个句子进行编号
    :param sent: 表示一个句子
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    读取之前存入文件中的word2id词典
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    补齐，将数据sequences内的word变成长度相同
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


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
    vocab_build('../train_test_data/word2id_bio.pkl', '../train_test_data/train_bio_word.txt', 10)
    with open('../train_test_data/word2id_bio.pkl', 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    for key, value in word2id.items():
        print(key, value)
    # raw_data = read_corpus('data_path/train_new.txt')
    # print(len(raw_data))


