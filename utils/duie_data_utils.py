# -*- coding: utf - 8 -*-

import re
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import BertWordPieceTokenizer
from utils.tokenization import FullTokenizer


def convert_raw_data_to_json_format(file_path, save_path):
    """将原始数据转成 JSON 格式

    转换后的文件格式:
        [
            {
                "text": "......",

                "spo_list": [
                    {
                        "subject": "...",
                        "subject_type": "...",
                        "object": "...",
                        "object_type": "...",
                        "predicate": "..."
                    }
                ]
            }
        ]

    Args:
        file_path: 原始数据文件路径
        save_path: JSON 格式文件存储路径

    Returns: None
    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        lines = reader.readlines()
    reader.close()

    examples = []
    for line in lines:
        item = json.loads(line)
        text = item['text']
        spo_list = item['spo_list']
        example = {
            'text': text,
            'spo_list': spo_list
        }
        examples.append(example)

    with open(save_path, mode='w', encoding='utf-8') as writer:
        writer.write(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def count_text_length(file_path):
    """统计 text 字段在 word piece 分词后的长度

    Args:
        file_path: 数据文件路径

    Returns: None

    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    tokenizer = BertWordPieceTokenizer('../bert-base-chinese/vocab.txt')

    text_lengths = []
    for item in data:
        text = item['text']
        tokenizer_output = tokenizer.encode(text)
        text_lengths.append(len(tokenizer_output.tokens))

    sorted_text_lengths = sorted(text_lengths)
    print(sorted_text_lengths[int(0.99 * len(sorted_text_lengths))])  # 163
    sns.displot(sorted_text_lengths)
    plt.show()


def show_schemas_info(file_path):
    """展示 schema 的相关统计信息, 具体包括:
        1. (subject type, object type) -> predicate 对应关系
        2. subject type 集合
        3. object type 集合
        4. predicate 集合
        5. subject type 和 object type 的并集
        6. subject type 和 object type 的交集

    Args:
        file_path: schemas 的路径

    Results:
        1. 一共存在 49 个不同的 predicate
        2. 一共存在 16 个不同的 subject type
        3. 一共存在 16 个不同的 object type
        4. subject type 和 object type 的并集中有 28 个元素
        5. subject type 和 object type 的交集中有 4 个元素
        6. (subject type, object type) -> [predicate] 对应关系共 36组
           其中每组对应的 predicate 的数量分别为:
           (2, 4, 1, 1, 2, 1, 1, 1, 1, 1, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    """
    schemas_dict = dict()
    subject_type_set = set()
    object_type_set = set()
    predicate_set = set()

    with open(file_path, mode='r', encoding='utf-8') as reader:
        lines = reader.readlines()
    reader.close()

    for line in lines:
        line = json.loads(line)

        subject_type = line['subject_type']
        object_type = line['object_type']
        predicate = line['predicate']

        subject_type_set.add(subject_type)
        object_type_set.add(object_type)
        if predicate in predicate_set:
            print(predicate)
        predicate_set.add(predicate)

        schemas_dict.setdefault(tuple((subject_type, object_type)), []).append(predicate)

    subject_type_and_object_type_set = subject_type_set | object_type_set
    subject_type_inter_object_type_set = subject_type_set & object_type_set
    print(f'schemas_dict: {len(schemas_dict)} {repr(schemas_dict)}')
    print(f'subject_type_set: {len(subject_type_set)} {repr(subject_type_set)}')
    print(f'object_type_set: {len(object_type_set)} {repr(object_type_set)}')
    print(f'predicate_set: {len(predicate_set)} {repr(predicate_set)}')
    print(f'subject_type_and_object_type_set: {len(subject_type_and_object_type_set)} {repr(subject_type_and_object_type_set)}')
    print(f'subject_type_inter_object_type_set: {len(subject_type_inter_object_type_set)} {repr(subject_type_inter_object_type_set)}')


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


def check_train_data(file_path):
    """

    Args:
        file_path:

    Returns:

    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    subject_type_set = set()
    object_type_set = set()
    predicate_set = set()
    triplet_set = set()
    total_invalid = 0

    for paragraph in data:
        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        for spo in spo_list:
            subject_type_set.add(spo['subject_type'])
            object_type_set.add(spo['object_type'])
            predicate_set.add(spo['predicate'])
            triplet_set.add(spo['subject_type'] + spo['predicate'] + spo['object_type'])

            if text.lower().find(spo['subject'].lower()) == -1 or text.lower().find(spo['object'].lower()) == -1:
                print(text, ' - ', spo['subject'], ' - ', spo['object'])
                total_invalid += 1

    print('total_invalid examples: ', total_invalid)

    origin_subject_types, origin_object_types, origin_predicates = load_schemas('../duie-dataset/schemas')
    origin_triplets = [
        subject_type + predicate + object_type
        for (subject_type, object_type, predicate) in zip(origin_subject_types, origin_object_types, origin_predicates)
    ]
    origin_subject_types = sorted(list(set(origin_subject_types)))
    origin_object_types = sorted(list(set(origin_object_types)))
    origin_predicates = sorted(list(set(origin_predicates)))
    origin_triplets = sorted(list(set(origin_triplets)))

    subject_types = sorted(list(subject_type_set))
    object_types = sorted(list(object_type_set))
    predicates = sorted(list(predicate_set))
    triplets = sorted(list(triplet_set))

    assert origin_subject_types == subject_types
    assert origin_object_types == object_types
    assert origin_predicates == predicates
    assert origin_triplets == triplets


def check_entity_overlap(file_path):
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    total_common_entity = 0
    total_overlap_entity = 0

    for paragraph in data:
        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        position_tuples = []
        for spo in spo_list:
            subject = spo['subject']
            object_ = spo['object']

            subject_start = text.lower().find(subject.lower())
            object_start = text.lower().find(object_.lower())

            position_tuples.append((object_start, object_start + len(object_) - 1))
            position_tuples.append((subject_start, subject_start + len(subject) - 1))

        position_tuples = sorted(position_tuples, key=lambda x: x[0])
        for i in range(len(position_tuples) - 1):
            if position_tuples[i][0] == position_tuples[i + 1][0] and position_tuples[i][1] == position_tuples[i + 1][1]:
                total_common_entity += 1
            elif position_tuples[i][1] > position_tuples[i + 1][0]:
                total_overlap_entity += 1

    print(total_common_entity)
    print(total_overlap_entity)


def remove_whitespace(file_path):
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    space_pattern = re.compile(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])')

    for paragraph_index, paragraph in enumerate(data):
        text = paragraph['text']
        spo_list = paragraph['spo_list']

        if re.match(space_pattern, text):
            print(re.match(space_pattern, text))

        for spo_index, spo in enumerate(spo_list):
            subject = spo['subject']
            object_ = spo['object']

            new_subject = ''
            spo_list[spo_index]['subject'] = new_subject

            new_object = ''
            spo_list[spo_index]['object'] = new_object
        data[paragraph_index]['spo_list'] = spo_list


if __name__ == '__main__':
    load_schemas('../duie-dataset/schemas')
    pass
