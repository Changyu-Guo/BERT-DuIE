# -*- coding: utf - 8 -*-

import tensorflow as tf
from transformers import TFBertModel


def create_model(num_labels, is_train=True, use_net_pretrain=False):

    inputs_ids = tf.keras.Input((None,), name='inputs_ids', dtype=tf.int32)
    inputs_mask = tf.keras.Input((None,), name='inputs_mask', dtype=tf.int32)
    segment_ids = tf.keras.Input((None,), name='segment_ids', dtype=tf.int32)

    if use_net_pretrain:
        core_bert = TFBertModel.from_pretrained('bert-base-chinese')
    else:
        core_bert = TFBertModel.from_pretrained('../bert-base-chinese')

    bert_output = core_bert({
        'input_ids': inputs_ids,
        'attention_mask': inputs_mask,
        'token_type_ids': segment_ids
    }, training=is_train)
