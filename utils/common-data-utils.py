# -*- coding: utf - 8 -*-

import json


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
    pass


if __name__ == '__main__':
    convert_raw_data_to_json_format(
        '../common-datasets/dev_data.json',
        '../common-datasets/dev_data_formatted.json'
    )
