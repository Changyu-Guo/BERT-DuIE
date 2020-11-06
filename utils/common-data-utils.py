# -*- coding: utf - 8 -*-

import json
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import BertWordPieceTokenizer


def convert_raw_data_to_json_format(file_path, save_path):
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

    origin_subject_types, origin_object_types, origin_predicates = load_schemas('../common-datasets/schemas')
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


if __name__ == '__main__':
    pass
