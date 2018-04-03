import logging
import sys
import argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity_keys(tag_seq, char_seq, keys):
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entities = []
    for key in keys:
        entities.append(get_entity_key(tag_seq, char_seq, key))
    return entities


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


# 将实体提取出来
def get_entity_one_(tag_seq, char_seq):
    sequence = []
    seq = ''
    for i, tag in enumerate(tag_seq):
        if tag == 'B' or tag == 'I':
            seq += char_seq[i]
        else:
            if len(seq) != 0:
                sequence.append(seq)
                seq = ''
    if len(seq) != 0:
        sequence.append(seq)
    return sequence


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

