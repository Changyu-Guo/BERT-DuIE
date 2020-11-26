# -*- coding: utf - 8 -*-

import os
import sys
import json
import shutil
from absl import flags, app
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel

sys.path.append('..')

from utils.distribution_utils import get_strategy_scope, get_distribution_strategy

FLAGS = flags.FLAGS


class Model(keras.Model):
    def __init__(self, output_units=49, drop_rate=0.1):
        super(Model, self).__init__()
        if os.path.exists('../bert-base-chinese'):
            self.core_bert = TFBertModel.from_pretrained('../bert-base-chinese')
        else:
            self.core_bert = TFBertModel.from_pretrained('bert-base-chinese')

        self.pooler_dropout = keras.layers.Dropout(
            name='pooler_dropout',
            rate=drop_rate
        )

        self.output_dense = keras.layers.Dense(
            name='output_dense',
            units=output_units,
            activation='sigmoid'
        )

    def call(self, inputs, training=True, mask=None):
        bert_outputs = self.core_bert(inputs, return_dict=True)
        pooler_output = bert_outputs['pooler_output']
        if training:
            pooler_output = self.pooler_dropout(pooler_output)
        output = self.output_dense(pooler_output)
        return output


def read_and_batch_tfrecord(
        file_path,
        max_seq_len,
        is_train,
        repeat,
        shuffle,
        batch_size
):
    name_to_features = {
        'example_indices': tf.io.FixedLenFeature([], tf.int64),
        'unique_ids': tf.io.FixedLenFeature([], tf.int64),
        'input_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([max_seq_len], tf.int64),
        'token_type_ids': tf.io.FixedLenFeature([max_seq_len], tf.int64)
    }

    if is_train:
        name_to_features['label_indices'] = tf.io.FixedLenFeature([49], tf.int64)

    dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    if repeat:
        dataset = dataset.repeat()

    def parse_example(example):
        parsed_example = tf.io.parse_single_example(example, name_to_features)
        return parsed_example

    dataset = dataset.map(
        parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def map_data_for_train(data):

    x = {
        'input_ids': data['input_ids'],
        'attention_mask': data['attention_mask'],
        'token_type_ids': data['token_type_ids']
    }

    y = data['label_indices']

    return x, y


def map_data_for_inference(data):

    x = {
        'input_ids': data['input_ids'],
        'attention_mask': data['attention_mask'],
        'token_type_ids': data['token_type_ids']
    }

    return x


def train(kwargs):

    reader = open(kwargs['train_json_for_train'], mode='r')
    train_meta = json.load(reader)
    reader.close()

    train_data_size = train_meta['data_size']
    steps_per_epoch = int(train_data_size // kwargs['batch_size'])
    if train_data_size % kwargs['batch_size'] != 0:
        steps_per_epoch += 1
    # total_train_steps = steps_per_epoch * kwargs['epochs']

    train_dataset = read_and_batch_tfrecord(
        file_path=kwargs['train_tfrecord_for_train'],
        max_seq_len=kwargs['max_seq_len'],
        is_train=True,
        repeat=True,
        shuffle=True,
        batch_size=kwargs['batch_size']
    )
    dev_dataset = read_and_batch_tfrecord(
        file_path=kwargs['dev_tfrecord_for_train'],
        max_seq_len=kwargs['max_seq_len'],
        is_train=True,
        repeat=False,
        shuffle=False,
        batch_size=kwargs['batch_size']
    )

    train_dataset = train_dataset.map(
        map_data_for_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dev_dataset = dev_dataset.map(
        map_data_for_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    distribution_strategy = get_distribution_strategy(
        distribution_strategy=kwargs['distribution_strategy'],
        num_gpus=1
    )
    with get_strategy_scope(distribution_strategy):

        model = Model()

        optimizer = keras.optimizers.Adam(learning_rate=kwargs['lr'])
        cost_function = keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(
            optimizer=optimizer,
            loss=cost_function,
            metrics=[keras.metrics.BinaryAccuracy()]
        )

        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=kwargs['log_save_dir']
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=3,
                restore_best_weights=True
            )
        ]

        model.fit(
            train_dataset,
            epochs=kwargs['epochs'],
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=dev_dataset
        )

        model.save_weights(kwargs['model_save_path'], save_format='h5')


def evaluate(kwargs):
    pass


def inference(kwargs):
    pass


def main(_):

    common_kwargs = {'version': 'version_' + str(FLAGS.version)}

    cur_version_path = 'data/version_' + str(FLAGS.version)
    cur_train_path = os.path.join(cur_version_path, 'train')
    cur_inference_path = os.path.join(cur_version_path, 'inference')

    # 获取文件路径
    for dataset in ['train', 'dev']:
        for suffix in ['tfrecord', 'pkl', 'json']:
            common_kwargs[dataset + '_' + suffix + '_for_train'] = os.path.join(
                cur_train_path,
                dataset + '.' + suffix
            )
            common_kwargs[dataset + '_' + suffix + '_for_inference'] = os.path.join(
                cur_inference_path,
                dataset + '.' + suffix
            )

    # 确定模型存储路径
    common_kwargs['model_save_path'] = os.path.join(cur_version_path, 'keras_model.h5')
    if os.path.exists(common_kwargs['model_save_path']):
        os.remove(common_kwargs['model_save_path'])

    # 确定日志存储路径
    common_kwargs['log_save_dir'] = os.path.join(cur_version_path, 'logs')
    if os.path.exists(common_kwargs['log_save_dir']):
        shutil.rmtree(common_kwargs['log_save_dir'])

    common_kwargs['distribution_strategy'] = FLAGS.distribution_strategy
    common_kwargs['epochs'] = FLAGS.epochs
    common_kwargs['batch_size'] = FLAGS.batch_size
    common_kwargs['max_seq_len'] = FLAGS.max_seq_len
    common_kwargs['lr'] = FLAGS.lr
    common_kwargs['enable_early_stopping'] = FLAGS.enable_early_stopping

    if FLAGS.run_train:
        train(common_kwargs)

    if FLAGS.run_evaluate:
        evaluate(common_kwargs)

    if FLAGS.run_inference:
        inference(common_kwargs)


if __name__ == '__main__':

    flags.DEFINE_boolean(
        name='run_train',
        default=False,
        help='Whether run train function.'
    )

    flags.DEFINE_boolean(
        name='run_evaluate',
        default=False,
        help='Whether run evaluate function.'
    )

    flags.DEFINE_boolean(
        name='run_inference',
        default=False,
        help='Whether run inference function.'
    )

    flags.DEFINE_integer(
        name='version',
        default=None,
        help='The current version.'
    )
    flags.mark_flag_as_required('version')

    flags.DEFINE_string(
        name='distribution_strategy',
        default=None,
        help='The distribution strategy.'
    )
    flags.mark_flag_as_required('distribution_strategy')

    flags.DEFINE_integer(
        name='epochs',
        default=None,
        help='Epochs'
    )
    flags.mark_flag_as_required('epochs')

    flags.DEFINE_integer(
        name='batch_size',
        default=None,
        help='The batch size in train and evaluate.'
    )
    flags.mark_flag_as_required('batch_size')

    flags.DEFINE_integer(
        name='max_seq_len',
        default=None,
        help='The max sequence length.'
    )
    flags.mark_flag_as_required('max_seq_len')

    flags.DEFINE_float(
        name='lr',
        default=None,
        help='The learning rate.'
    )
    flags.mark_flag_as_required('lr')

    flags.DEFINE_boolean(
        name='enable_early_stopping',
        default=None,
        help='Whether enable early stopping.'
    )
    flags.mark_flag_as_required('enable_early_stopping')

    app.run(main)
