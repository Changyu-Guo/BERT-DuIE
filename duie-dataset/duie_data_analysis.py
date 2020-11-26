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


def check_entity_overlap_within_context(file_path):
    """检查实体重叠情况, 具体包括:
        1. 统计出现 subject 实体相同的 context 条目
        2. 统计出现 object 实体相同的 context 条目
        3. 统计同时出现 subject 实体和 object 实体相同的 context 条目
        4. 统计出现 subject 实体或 object 实体相同的 context 条目
        5. 统计出现 subject 实体重叠的 context 条目
        6. 统计出现 object 实体重叠的 context 条目
        7. 统计同时出现 subject 实体和 object 实体重叠的 context 条目
        8. 统计出现 subject 实体或 object 实体重叠的 context 条目

    Args:
        file_path: 训练集或验证集的文件路径

    Results:
        训练集上统计结果:
            0. total: 173108
            1. subject common: 87599
            2. object common: 16909
            3. subject & object common: 11835
            4. subject | object common: 92673
            5. subject overlap: 1164
            6. object overlap: 2413
            7. subject & object overlap: 5
            8. subject | object overlap: 3572
        验证集上统计结果:
            0. total: 21639
            1. subject common: 10987
            2. object common: 2075
            3. subject & object common: 1435
            4. subject | object common: 11627
            5. subject overlap: 137
            6. object overlap: 301
            7. subject & object overlap: 0
            8. subject | object overlap: 438
    """
    with open(file_path, mode='r', encoding='utf-8') as reader:
        data = json.load(reader)
    reader.close()

    total_contexts = 0
    total_contexts_with_common_subject_entity = 0
    total_contexts_with_common_object_entity = 0
    total_contexts_with_common_subject_and_object_entity = 0
    total_contexts_with_common_subject_or_object_entity = 0
    total_contexts_with_overlap_subject_entity = 0
    total_contexts_with_overlap_object_entity = 0
    total_contexts_with_overlap_subject_and_object_entity = 0
    total_contexts_with_overlap_subject_or_object_entity = 0

    for paragraph in data:

        total_contexts += 1

        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        context_subject_entity_list = []
        context_object_entity_list = []
        subject_position_tuples = []
        object_position_tuples = []
        for spo in spo_list:

            subject = spo['subject']
            object_ = spo['object']

            # 当前 context 中所有的 subject entity
            context_subject_entity_list.append(subject)
            # 当前 context 中所有的 object entity
            context_object_entity_list.append(object_)

            subject_start = text.lower().find(subject.lower())
            object_start = text.lower().find(object_.lower())
            if subject_start == -1 or object_start == -1:
                raise ValueError('Cannot find an answer of: ', text, subject, object_)

            # 当前 context 中所有 subject entity 的 **起始位置**
            subject_position_tuples.append((subject_start, subject_start + len(subject) - 1))
            # 当前 context 中所有 object entity 的 **起始位置**
            object_position_tuples.append((object_start, object_start + len(object_) - 1))

        # subject entity 出现重复
        if len(set(context_subject_entity_list)) != len(context_subject_entity_list):
            total_contexts_with_common_subject_entity += 1
        # object entity 出现重复
        if len(set(context_object_entity_list)) != len(context_object_entity_list):
            total_contexts_with_common_object_entity += 1
        # subject entity 和 object entity 同时出现重复
        if len(set(context_subject_entity_list)) != len(context_subject_entity_list) and \
                len(set(context_object_entity_list)) != len(context_object_entity_list):
            total_contexts_with_common_subject_and_object_entity += 1
        # subject entity 或 object entity 出现重复
        if len(set(context_subject_entity_list)) != len(context_subject_entity_list) or \
                len(set(context_object_entity_list)) != len(context_object_entity_list):
            total_contexts_with_common_subject_or_object_entity += 1

        subject_position_tuples = sorted(list(set(subject_position_tuples)), key=lambda x: x[0])
        object_position_tuples = sorted(list(set(object_position_tuples)), key=lambda x: x[0])
        subject_overlap_position_tuples = set()
        object_overlap_position_tuples = set()

        # subject 出现重叠
        for i in range(len(subject_position_tuples) - 1):
            if subject_position_tuples[i][1] > subject_position_tuples[i + 1][0]:
                total_contexts_with_overlap_subject_entity += 1
                subject_overlap_position_tuples.add((subject_position_tuples[i], subject_position_tuples[i + 1]))
                break

        # object 出现重叠
        for i in range(len(object_position_tuples) - 1):
            if object_position_tuples[i][1] > object_position_tuples[i + 1][0]:
                total_contexts_with_overlap_object_entity += 1
                object_overlap_position_tuples.add((object_position_tuples[i], object_position_tuples[i + 1]))
                break

        total_contexts_with_overlap_subject_and_object_entity += len(
            subject_overlap_position_tuples & object_overlap_position_tuples
        )
        total_contexts_with_overlap_subject_or_object_entity += len(
            subject_overlap_position_tuples | object_overlap_position_tuples
        )

    print('total contexts: ', total_contexts)
    print('total contexts with common subject entity: ', total_contexts_with_common_subject_entity)
    print('total contexts with common object entity: ', total_contexts_with_common_object_entity)
    print('total contexts with common subject and object entity: ', total_contexts_with_common_subject_and_object_entity)
    print('total contexts with common subject or object entity: ', total_contexts_with_common_subject_or_object_entity)
    print('total contexts with overlap subject entity: ', total_contexts_with_overlap_subject_entity)
    print('total contexts with overlap object entity: ', total_contexts_with_overlap_object_entity)
    print('total contexts with overlap subject and object entity: ', total_contexts_with_overlap_subject_and_object_entity)
    print('total contexts with overlap subject or object entity: ', total_contexts_with_overlap_subject_or_object_entity)


def check_entity_type_consistency_with_common_entity(file_path):
    """检查在同一 context 下面, 所有相同的 subject 或 object, 它们的 type 是否一致

    Args:
        file_path: 训练集或测试集的路径

    Results:
        训练集上统计结果:
            一个 subject entity 对应多个 entity type 的 context 条目: 28467
                出现最多的情况: 书籍-图书作品:12675/图书作品-网络小说:12174
            一个 object entity 对应多个 entity type 的 context 条目: 887
                出现最多的情况:国家-地点:644/Text-国家:94
        验证集上统计结果:
            一个 subject entity 对应多个 entity type 的 context 条目: 3711
                出现最多的情况: 书籍-图书作品:1663/图书作品-网络小说:1527
            一个 object entity 对应多个 entity type 的 context 条目: 108
                出现最多的情况: 国家-地点:79/Text-国家:15
    Notes:
        根据统计结果得出, 数据集中存在大量的相同实体具有不同类型的情况
        经初步分析, 直接对 context 进行标注(标注出实体类型, 包括 token-labeling 和 span-labeling), 然后判断实体间关系的方法行不通
        变通方法是在 span-labeling 中, 对每个 span 进行一个 **多标签分类任务** 而不是一个 **单类别多分类任务**
        但具体实验结果还有待进一步验证(由于任务更加复杂有可能会变差)
        而对于 token-labeling, 初步想法是对每个 token 进行一个多类别的标注
        这种方法跟 span-labeling 类似, 具体结果有待验证
    """
    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    total_subject_inconsistency = 0
    total_object_inconsistency = 0
    common_subject_entity_type_counter = collections.defaultdict(int)
    common_object_entity_type_counter = collections.defaultdict(int)

    for paragraph in data:

        spo_list = paragraph['spo_list']

        subject2subject_type = collections.defaultdict(list)
        object2object_type = collections.defaultdict(list)

        for spo in spo_list:

            subject = spo['subject']
            object_ = spo['object']
            subject_type = spo['subject_type']
            object_type = spo['object_type']

            subject2subject_type[subject].append(subject_type)
            object2object_type[object_].append(object_type)

        for key, value in subject2subject_type.items():
            value = set(value)
            if len(value) > 1:
                common_subject_entity_type_counter['-'.join(sorted(value))] += 1
                total_subject_inconsistency += 1

        for key, value in object2object_type.items():
            value = set(value)
            if len(value) > 1:
                common_object_entity_type_counter['-'.join(sorted(value))] += 1
                total_object_inconsistency += 1

    print('total subject inconsistency: ', total_subject_inconsistency)
    print('total object inconsistency: ', total_object_inconsistency)
    print('common subject entity types: ', dict(sorted(common_subject_entity_type_counter.items(), key=lambda x: x[1], reverse=True)))
    print('common object entity types: ', dict(sorted(common_object_entity_type_counter.items(), key=lambda x: x[1], reverse=True)))


def check_entity_overlap_with_predicate_constraint(file_path):
    """假定已经抽取到 predicate 的前提下, 检查如下 common 和 overlap 的情况:
        1. 给定 predicate 的前提下, subject 是否出现相同
        2. 给定 predicate 的前提下, object 是否出现相同
        3. 给定 predicate 的前提下, subject 是否出现 overlap
        4. 给定 predicate 的前提下, object 是否出现 overlap
        # Notice 在 predicate 和 subject/object 给定的情况下, object/subject 不可能出现完全相同的情况
        5. 给定 predicate 和 subject 的前提下, object 是否出现 overlap
        6. 给定 predicate 和 object 的前提下, subject 是否出现 overlap

    Args:
        file_path: 训练集或验证集文件路径

    Results:
        训练集上统计结果:
            0. total examples:
            1. subject overlap:
            2. object overlap:
            3. object overlap | subject:
            4. subject overlap | object:
        验证集上统计结果:1
            1. subject overlap:
            2. object overlap:
            3. object overlap | subject:
            4. subject overlap | object:
    """
    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    total_examples = 0
    total_subject_entity_overlap_examples = 0
    total_object_entity_overlap_examples = 0
    total_subject_object_entity_overlap_examples = 0
    total_object_subject_entity_overlap_examples = 0

    for paragraph in data:

        text: str = paragraph['text']
        spo_list = paragraph['spo_list']

        predicate_subject_position_tuples = collections.defaultdict(list)
        predicate_object_position_tuples = collections.defaultdict(list)
        predicate_subject_object_position_tuples = collections.defaultdict(list)
        predicate_object_subject_position_tuples = collections.defaultdict(list)

        for spo in spo_list:

            total_examples += 1

            subject = spo['subject']
            subject_type = spo['subject_type']
            object_ = spo['object']
            object_type = spo['object']
            predicate = spo['predicate']

            subject_start = text.lower().find(subject.lower())
            object_start = text.lower().find(object_.lower())
            if subject_start == -1 or object_start == -1:
                raise ValueError('cannot find an answer of: ', text, subject, object_)

            predicate_subject_position_tuples[predicate + subject_type].append(
                (subject_start, subject_start + len(subject) - 1)
            )
            predicate_object_position_tuples[predicate + object_type].append(
                (object_start, object_start + len(object_) - 1)
            )

            predicate_subject_object_position_tuples[predicate + subject + object_type].append(
                (object_start, object_start + len(object_) - 1)
            )
            predicate_object_subject_position_tuples[predicate + object_ + subject_type].append(
                (subject_start, subject_start + len(subject) - 1)
            )

        # 在 predicate 给定的条件下, subject overlap 的样本数量
        for key, value in predicate_subject_position_tuples.items():
            value = sorted(list(set(value)), key=lambda x: x[0])
            for i in range(len(value) - 1):
                if value[i][1] > value[i + 1][0]:
                    total_subject_entity_overlap_examples += 1
        # 在 predicate 给定的条件下, object overlap 的样本数量
        for key, value in predicate_object_position_tuples.items():
            value = sorted(list(set(value)), key=lambda x: x[0])
            for i in range(len(value) - 1):
                if value[i][1] > value[i + 1][0]:
                    total_object_entity_overlap_examples += 1

        # 在已知 predicate 和 subject 的前提下, object overlap 的样本数量
        for key, value in predicate_subject_object_position_tuples.items():
            value = sorted(list(set(value)), key=lambda x: x[0])
            for i in range(len(value) - 1):
                if value[i][1] > value[i + 1][0]:
                    total_subject_object_entity_overlap_examples += 1
        # 在已知 predicate 和 object 的前提下, subject overlap 的样本数量
        for key, value in predicate_object_subject_position_tuples.items():
            value = sorted(list(set(value)), key=lambda x: x[0])
            for i in range(len(value) - 1):
                if value[i][1] > value[i + 1][0]:
                    total_object_subject_entity_overlap_examples += 1

    print('total examples: ', total_examples)
    print('total subject entity overlap examples: ', total_subject_entity_overlap_examples)
    print('total object entity overlap examples: ', total_object_entity_overlap_examples)
    print('total subject object entity overlap examples: ', total_subject_object_entity_overlap_examples)
    print('total object subject entity overlap examples: ', total_object_subject_entity_overlap_examples)


def check_pieces_common(entity_pieces, context_pieces):
    """检查 entity pieces 是否能够连续的在 context pieces 中找到
    Args:
        entity_pieces: 命名实体的 word piece
        context_pieces: 上下文的 word piece
    """
    indices = [i for i, x in enumerate(context_pieces) if x == entity_pieces[0]]
    for start_pos in indices:
        end_pos = start_pos + len(entity_pieces)
        candidate_pieces = context_pieces[start_pos: end_pos]
        common = [
            entity_piece == context_piece
            for entity_piece, context_piece in zip(
                entity_pieces,
                candidate_pieces
            )
        ]
        if all(common):
            return True

    return False


def check_answer_recovery_after_word_piece(file_path):
    """检查在对 context 和 entity 都做 word piece 之后, entity 是否还能在原文中找到
    Args:
        file_path: 训练集或验证集路径

    Results:
        训练集上统计结果:
            subject: 19
            object: 122
        验证集上统计结果:
            subject: 1
            object: 14
    """
    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    tokenizer = BertWordPieceTokenizer('../bert-base-chinese/vocab.txt')

    total_fail_subject_check = 0
    total_fail_object_check = 0

    for paragraph in data:

        text = paragraph['text']
        text_pieces = tokenizer.encode(text, add_special_tokens=False).tokens

        spo_list = paragraph['spo_list']

        for spo in spo_list:

            subject = spo['subject']
            object_ = spo['object']

            subject_pieces = tokenizer.encode(subject, add_special_tokens=False).tokens
            object_pieces = tokenizer.encode(object_, add_special_tokens=False).tokens

            if not check_pieces_common(subject_pieces, text_pieces):
                print(subject_pieces, text)
                total_fail_subject_check += 1
            if not check_pieces_common(object_pieces, text_pieces):
                print(object_pieces, text)
                total_fail_object_check += 1

    print('total fail subject check: ', total_fail_subject_check)
    print('total fail object check: ', total_fail_object_check)


def fix_answer_recovery(file_path):
    """对进行 word piece 之后不能将答案恢复的样本进行修复
    Args:
        file_path: 训练集或验证集的路径

    Results:
        目前仅通过在实体前后插入空格就解决了相应的问题, 因此没有进行进一步处理
    """
    def insert_white_space_both_entity_sides(entity: str, text: str):
        start_pos = text.lower().find(entity.lower())
        if start_pos != -1:
            text = text[:start_pos] + ' ' + entity + ' ' + text[start_pos + len(entity):]
            return text
        else:
            raise ValueError(text, ' can not found ', entity)

    reader = open(file_path, mode='r', encoding='utf-8')
    data = json.load(reader)
    reader.close()

    tokenizer = BertWordPieceTokenizer('../bert-base-chinese/vocab.txt')

    for paragraph_index, paragraph in enumerate(data):

        text = paragraph['text']
        text_pieces = tokenizer.encode(text, add_special_tokens=False).tokens
        spo_list = paragraph['spo_list']

        do_change_text = False

        for spo in spo_list:

            subject = spo['subject']
            object_ = spo['object']

            subject_pieces = tokenizer.encode(subject, add_special_tokens=False).tokens
            object_pieces = tokenizer.encode(object_, add_special_tokens=False).tokens

            if not check_pieces_common(subject_pieces, text_pieces):
                text = insert_white_space_both_entity_sides(subject, text)
                # text change, redo tokenizer encode
                text_pieces = tokenizer.encode(text, add_special_tokens=False).tokens
                do_change_text = True
            if not check_pieces_common(object_pieces, text_pieces):
                text = insert_white_space_both_entity_sides(object_, text)
                # text change, redo tokenizer encode
                text_pieces = tokenizer.encode(text, add_special_tokens=False).tokens
                do_change_text = True

        if do_change_text:
            data[paragraph_index]['text'] = text

    writer = open(file_path, mode='w', encoding='utf-8')
    writer.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    writer.close()


def count_text_length(file_path):
    """统计 text 字段在 word piece 分词后的长度

    Args:
        file_path: 数据文件路径

    Results:
        训练集上统计结果:
            max text length: 299
            min text length: 6
            avg text length: 52
            0.95 text length: 110
            0.99 text length: 163
        验证集上统计结果:
            max text length: 300
            min text length: 7
            avg text length: 53
            0.95 text length: 111
            0.99 text length: 163
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
    print('max: ', max(sorted_text_lengths))
    print('min: ', min(sorted_text_lengths))
    print('avg: ', sum(sorted_text_lengths) / len(sorted_text_lengths))
    print('0.95: ', sorted_text_lengths[int(0.95 * len(sorted_text_lengths))])
    print('0.99: ', sorted_text_lengths[int(0.99 * len(sorted_text_lengths))])
    sns.displot(sorted_text_lengths)
    plt.show()


if __name__ == '__main__':
    show_schemas_info('schemas')
    pass
