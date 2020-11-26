# -*- coding: utf - 8 -*-

import json


def load_schemas(file_path):
    """加载 schemas

    Args:
        file_path: schemas 文件路径

    Returns:
        subject_type_list: list of subject type
        object_type_list: list of object type
        predicate_list: list of predicate
    """
    subject_type_list = []
    object_type_list = []
    predicate_list = []

    with open(file_path, mode='r', encoding='utf-8') as reader:
        lines = reader.readlines()
    reader.close()

    for line in lines:
        line = json.loads(line)
        subject_type_list.append(line['subject_type'])
        object_type_list.append(line['object_type'])
        predicate_list.append(line['predicate'])

    return subject_type_list, object_type_list, predicate_list
