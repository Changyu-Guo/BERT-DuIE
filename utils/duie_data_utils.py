# -*- coding: utf - 8 -*-

import re
import sys
import json
import collections
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
    """检查训练数据和验证数据, 其中包括如下指标:
        1. 所有 subject type
        2. 所有 object type
        3. 所有 predicate
        4. (subject type, predicate, object type) 三元组
        5. 每种不同的 (subject type, predicate, object type) 三元组对应的样本数目
        6. 在同一 context 下出现重复 predicate 的 context 数目
        7. 在原文中找不到 subject 或 object 的样本数目

    Args:
        file_path: 训练集或测试集的路径

    Results:
        训练集结果:
            1. 所有 subject type 共有 16 种, 和 schemas 中所有 subject type 相同
            2. 所有 object type 共有 16 种, 和 schemas 中所有 object type 相同
            3. 所有 predicate 共有 49 种, 和 schemas 中所有 predicate 相同
            4. 所有 triplet 共有 50 种, 和 schemas 中所有 triplet 相同
            5. 每种不同的三元组对应的样本数目为:
                   59893, 10342, 2905, 23290, 12261, 12877, 33853, 25605, 383, 19813
                   22007, 920, 3433, 5462, 5462, 11453, 11515, 6667, 6419, 4701, 17709
                   9046, 9607, 3516, 9107, 2733, 1551, 10402, 575, 2466, 1511, 3522
                   397, 3799, 633, 1492, 553, 1160, 397, 409, 660, 250, 502, 1368, 1008, 368
                   145, 25, 26, 20
            6. 在同一 context 下出现重复 predicate 的 context 数目为 28188, 占比 0.16
            7. 共有 9 条在原文中找不到答案的样本

        验证集结果:
            1. 所有 subject type 共有 16 种, 和 schemas 中所有 subject type 相同
            2. 所有 object type 共有 16 种, 和 schemas 中所有 object type 相同
            3. 所有 predicate 共有 49 种, 和 schemas 中所有 predicate 相同
            4. 所有 triplet 共有 50 种, 和 schemas 中所有 triplet 相同
            5. 每种不同的三元组对应的样本数目为:
                   2371, 2894, 3085, 1123, 851, 2698, 680, 680, 4395, 2273, 7414, 1655
                   1436, 1271, 454, 1658, 1287, 393, 866, 425, 1433, 1119, 339, 81, 197
                   419, 609, 171, 463, 1154, 77, 71, 43, 138, 63, 61, 181, 310, 144, 79
                   45, 47, 208, 50, 39, 103, 2, 3, 15, 4
            6. 在同一 context 下出现重复 predicate 的 context 数目为 3556, 占比 0.16
            7. 不存在在原文中找不到答案的样本
    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    total_example = 0
    subject_type_set = set()
    object_type_set = set()
    predicate_set = set()
    triplet_set = set()
    triplet_counter = collections.defaultdict(int)
    predicate_with_multi_answer = 0
    total_invalid = 0

    for paragraph in data:
        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        spo_predicates = []
        for spo in spo_list:
            subject_type_set.add(spo['subject_type'])
            object_type_set.add(spo['object_type'])
            predicate_set.add(spo['predicate'])
            triplet_set.add(spo['subject_type'] + spo['predicate'] + spo['object_type'])
            triplet_counter[spo['subject_type'] + spo['predicate'] + spo['object_type']] += 1
            spo_predicates.append(spo['predicate'])

            if text.lower().find(spo['subject'].lower()) == -1 or text.lower().find(spo['object'].lower()) == -1:
                print(text, ' - ', spo['subject'], ' - ', spo['object'])
                total_invalid += 1
        if len(set(spo_predicates)) != len(spo_predicates):
            predicate_with_multi_answer += 1
        total_example += 1

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

    print('len subject types: ', len(subject_types))
    print('len object types: ', len(object_types))
    print('len predicates: ', len(predicates))
    print('len triplets: ', len(triplets))
    print('predicate with multi answer: ', predicate_with_multi_answer)
    print('total example: ', total_example)

    temp_list = []
    for key, value in triplet_counter.items():
        temp_list.append(value)
    print(temp_list)
    sns.barplot(x=list(range(len(temp_list))), y=temp_list)
    plt.show()


def fix_item_without_answer():
    """
        由于没有答案的样本只有 9 条, 因此使用了人工修复的方式
        1. 将答案编辑为原文中存在的 (6条)
        2. 不能编辑回去的直接删掉 (3条)
    """
    pass


def check_entity_overlap(file_path):
    """检查实体重叠情况, 具体包括:
        1. 统计出现相同实体的 context 条目
        2. 统计出现实体重叠的 context 条目(不包含相同实体的条目)

    Args:
        file_path: 训练集或验证集的文件路径

    Results:
        训练集上统计结果:
            1. common: 92673, 占比 0.25
            2. overlap: 5852, 占比 0.016
        验证集上统计结果:
            1. common: 11627, 占比 0.26
            2. overlap: 697, 占比 0.015
    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    total_examples = 0
    total_examples_with_common_entity = 0
    total_examples_with_overlap_entity = 0

    for paragraph in data:
        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        position_tuples = []
        context_subject_entity_list = []
        context_object_entity_list = []
        for spo in spo_list:

            total_examples += 1

            subject = spo['subject']
            object_ = spo['object']

            context_subject_entity_list.append(subject)
            context_object_entity_list.append(object_)

            subject_start = text.lower().find(subject.lower())
            object_start = text.lower().find(object_.lower())
            if subject_start == -1 or object_start == -1:
                raise ValueError('cannot find an answer of: ', text, subject, object_)

            position_tuples.append((subject_start, subject_start + len(subject) - 1))
            position_tuples.append((object_start, object_start + len(object_) - 1))

        if len(set(context_subject_entity_list)) != len(context_subject_entity_list) or \
                len(set(context_object_entity_list)) != len(context_object_entity_list):
            total_examples_with_common_entity += 1

        position_tuples = sorted(list(set(position_tuples)), key=lambda x: x[0])
        for i in range(len(position_tuples) - 1):
            if position_tuples[i][1] > position_tuples[i + 1][0]:
                total_examples_with_overlap_entity += 1
                break

    print('total examples: ', total_examples)
    print('total examples with common entity: ', total_examples_with_common_entity)
    print('total examples with overlap entity: ', total_examples_with_overlap_entity)


def check_entity_overlap_with_predicate_constraint(file_path):
    """假定已经抽取到 predicate 的前提下, 检查如下 common 和 overlap 的情况:
        1. 给定 predicate 的前提下, subject 是否出现相同, 如果有, 则不能使用预测开始和结束位置的方法
        2. 给定 predicate 的前提下, object 是否出现相同, 如果有, 则不能使用预测开始和结束位置的方法
        3. 给定 predicate 和 subject 的前提下, object 是否出现相同
        4. 给定 predicate 和 object 的前提下, subject 是否出现相同
        5. 给定 predicate 的前提下, subject 是否出现 overlap
        6. 给定 predicate 的前提下, object 是否出现 overlap
        7. 给定 predicate 和 subject 的前提下, object 是否出现 overlap
        8. 给定 predicate 和 object 的前提下, subject 是否出现 overlap

    Args:
        file_path: 训练集或验证集文件路径

    Results:
        训练集上统计结果:

    """
    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    total_examples = 0
    total_subject_entity_common_examples = 0
    total_object_entity_common_examples = 0
    total_subject_object_common_examples = 0
    total_object_subject_common_examples = 0

    for paragraph in data:
        spo_list = paragraph['spo_list']

        predicate_subject_entity = collections.defaultdict(int)
        predicate_object_entity = collections.defaultdict(int)
        predicate_subject_object_entity = collections.defaultdict(int)
        predicate_object_subject_entity = collections.defaultdict(int)
        for spo in spo_list:

            total_examples += 1

            subject = spo['subject']
            object_ = spo['object']
            predicate = spo['predicate']

            predicate_subject_entity[predicate + subject] += 1
            predicate_object_entity[predicate + object_] += 1
            predicate_subject_object_entity[predicate + subject + object_] += 1
            predicate_object_subject_entity[predicate + object_ + subject] += 1

        for key, value in predicate_subject_entity.items():
            if value > 1:
                total_subject_entity_common_examples += 1

        for key, value in predicate_object_entity.items():
            if value > 1:
                total_object_entity_common_examples += 1

        for key, value in predicate_subject_object_entity.items():
            if value > 1:
                total_subject_object_common_examples += 1

        for key, value in predicate_object_subject_entity.items():
            if value > 1:
                total_object_subject_common_examples += 1

    print('total subject entity common examples: ', total_subject_entity_common_examples)
    print('total object entity common examples: ', total_object_entity_common_examples)
    print('total subject object overlap examples: ', total_subject_object_common_examples)
    print('total object subject overlap examples: ', total_object_subject_common_examples)


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
    check_entity_overlap_with_predicate_constraint('../duie-dataset/dev_data_formatted.json')
    pass
