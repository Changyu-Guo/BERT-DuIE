# -*- coding: utf - 8 -*-

import os
import sys
import json
import gzip
import pickle
import collections
import tensorflow as tf
from absl import flags, app
from tokenizers import BertWordPieceTokenizer
from utils.duie_data_utils import load_schemas

FLAGS = flags.FLAGS


class Example:
    def __init__(self, paragraph_index, text, predicates):
        self.paragraph_index = paragraph_index
        self.text = text
        self.predicates = predicates


class Feature:
    def __init__(
            self,
            example_index,
            unique_id,
            input_ids,
            attention_mask,
            token_type_ids,
            label_indices=None
    ):
        self.example_index = example_index
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_indices = label_indices


class FeatureWriter:

    def __init__(self, filename, is_train):

        self.filename = filename
        self.is_train = is_train

        options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._writer = tf.io.TFRecordWriter(filename, options=options)

        self.num_features = 0

    def process_feature(self, feature):

        self.num_features += 1

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features['example_indices'] = create_int_feature([feature.example_index])
        features['unique_ids'] = create_int_feature([feature.unique_id])
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['attention_mask'] = create_int_feature(feature.attention_mask)
        features['token_type_ids'] = create_int_feature(feature.token_type_ids)

        if self.is_train:
            features['label_indices'] = create_int_feature(feature.label_indices)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(example.SerializeToString())

    def close(self):
        self._writer.close()


def generate_formatted_data(origin_file_path, save_path):
    reader = open(origin_file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    writer = open(save_path, mode='w', encoding='utf-8')
    writer.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def read_examples(file_path, is_train):
    """从原始数据集中读取出所有的 example
    Args:
        file_path: 训练集或验证集路径
        is_train: 是否是生成训练使用的数据, 如果是, 则包含 predicates, 不是的话则不包含

    Returns: 所有的 example
    """

    # ########### Load Data ############
    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    # ########### Extract Examples ###############
    examples = []
    for paragraph_index, paragraph in enumerate(data):

        text = paragraph['text']
        spo_list = paragraph['spo_list']

        predicates = None

        if is_train:

            predicates = set()

            for spo in spo_list:
                predicates.add(spo['predicate'])

            predicates = list(predicates)

        example = Example(
            paragraph_index=paragraph_index,
            text=text,
            predicates=predicates
        )
        examples.append(example)

    return examples


def convert_examples_to_features(examples, max_seq_len, output_fn, is_train):
    """将 examples 处理成 features
    Args:
        examples: 所有样本
        max_seq_len: 特征的最大长度
        output_fn: 特征处理函数
        is_train: 是否是生成训练特征

    Returns: None
    """

    # ########## Load Tokenizer #######
    tokenizer = BertWordPieceTokenizer('../bert-base-chinese/vocab.txt')
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_length=max_seq_len)

    # ########## Load Schemas ########
    _, _, predicates = load_schemas('../duie-dataset/schemas')
    predicates = list(set(predicates))
    predicate_to_index = {}
    for index, predicate in enumerate(predicates):
        predicate_to_index[predicate] = index

    # ########## Convert #########
    base_id = 1000000000
    unique_id = base_id
    for example_index, example in enumerate(examples):

        text = example.text
        tokenizer_out = tokenizer.encode(text)

        label_indices = None
        if is_train:

            text_predicates = example.predicates

            label_indices = [0] * 49
            for predicate in text_predicates:
                label_indices[predicate_to_index[predicate]] += 1

        feature = Feature(
            example_index=example_index,
            unique_id=unique_id,
            input_ids=tokenizer_out.ids,
            attention_mask=tokenizer_out.attention_mask,
            token_type_ids=tokenizer_out.type_ids,
            label_indices=label_indices
        )

        output_fn(feature)

        unique_id += 1


def generate_tfrecord(
        input_file_path,
        tfrecord_save_path,
        features_save_path,
        meta_save_path,
        max_seq_len,
        is_train
):

    writer = FeatureWriter(filename=tfrecord_save_path, is_train=is_train)

    examples = read_examples(input_file_path, is_train=is_train)

    features = []

    def append_feature(feature):
        features.append(feature)
        writer.process_feature(feature)

    convert_examples_to_features(
        examples=examples,
        max_seq_len=max_seq_len,
        output_fn=append_feature,
        is_train=is_train
    )

    meta_data = {
        'data_size': writer.num_features,
        'max_seq_len': max_seq_len
    }
    writer.close()

    writer = open(meta_save_path, mode='w')
    writer.write(json.dumps(meta_data, ensure_ascii=False, indent=2) + '\n')
    writer.close()

    writer = gzip.open(features_save_path, mode='wb')
    pickle.dump(features, writer, protocol=pickle.HIGHEST_PROTOCOL)
    writer.close()


def main(_):

    cur_version = 'version_' + str(FLAGS.version)

    cur_version_dir = os.path.join('data', cur_version)
    if not os.path.exists(cur_version_dir):
        os.mkdir(cur_version_dir)

    input_train_file_path = os.path.join(cur_version_dir, 'train.json')
    input_dev_file_path = os.path.join(cur_version_dir, 'dev.json')

    generate_formatted_data(
        origin_file_path=FLAGS.origin_train_file_path,
        save_path=input_train_file_path
    )
    generate_formatted_data(
        origin_file_path=FLAGS.origin_dev_file_path,
        save_path=input_dev_file_path
    )

    train_files_dir = os.path.join(cur_version_dir, 'train')
    if not os.path.exists(train_files_dir):
        os.mkdir(train_files_dir)

    inference_files_dir = os.path.join(cur_version_dir, 'inference')
    if not os.path.exists(inference_files_dir):
        os.mkdir(inference_files_dir)

    file_suffixes = ['tfrecord', 'pkl', 'json']
    train_file_paths = collections.OrderedDict()
    inference_file_paths = collections.OrderedDict()

    for file_suffix in file_suffixes:

        train_file_paths['train_' + file_suffix] = os.path.join(train_files_dir, 'train.' + file_suffix)
        train_file_paths['dev_' + file_suffix] = os.path.join(train_files_dir, 'dev.' + file_suffix)

        inference_file_paths['train_' + file_suffix] = os.path.join(inference_files_dir, 'train.' + file_suffix)
        inference_file_paths['dev_' + file_suffix] = os.path.join(inference_files_dir, 'dev.' + file_suffix)

    # generate train files for train
    generate_tfrecord(
        input_file_path=input_train_file_path,
        tfrecord_save_path=train_file_paths['train_tfrecord'],
        features_save_path=train_file_paths['train_pkl'],
        meta_save_path=train_file_paths['train_json'],
        max_seq_len=FLAGS.max_seq_len,
        is_train=True
    )

    # generate dev files for train
    generate_tfrecord(
        input_file_path=input_dev_file_path,
        tfrecord_save_path=train_file_paths['dev_tfrecord'],
        features_save_path=train_file_paths['dev_pkl'],
        meta_save_path=train_file_paths['dev_json'],
        max_seq_len=FLAGS.max_seq_len,
        is_train=True
    )

    # generate train files for inference
    generate_tfrecord(
        input_file_path=input_train_file_path,
        tfrecord_save_path=inference_file_paths['train_tfrecord'],
        features_save_path=inference_file_paths['train_pkl'],
        meta_save_path=inference_file_paths['train_json'],
        max_seq_len=FLAGS.max_seq_len,
        is_train=False
    )

    # generate dev files for inference
    generate_tfrecord(
        input_file_path=input_dev_file_path,
        tfrecord_save_path=inference_file_paths['dev_tfrecord'],
        features_save_path=inference_file_paths['dev_pkl'],
        meta_save_path=inference_file_paths['dev_json'],
        max_seq_len=FLAGS.max_seq_len,
        is_train=False
    )


if __name__ == '__main__':

    flags.DEFINE_string(
        name='origin_train_file_path',
        default=None,
        help='The origin train file path.'
    )
    flags.mark_flag_as_required('origin_train_file_path')

    flags.DEFINE_string(
        name='origin_dev_file_path',
        default=None,
        help='The origin dev file path.'
    )
    flags.mark_flag_as_required('origin_dev_file_path')

    flags.DEFINE_integer(
        name='version',
        default=None,
        help='The version of current run.'
    )

    flags.DEFINE_integer(
        name='max_seq_len',
        default=None,
        help='Max seq length.'
    )
    flags.mark_flag_as_required('max_seq_len')

    app.run(main)
