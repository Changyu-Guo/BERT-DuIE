# -*- coding: utf - 8 -*-

import json
from tokenizers import BertWordPieceTokenizer


class DataProcessor:
    def __init__(
            self,
            vocab_file_path='../bert-base-chinese/vocab.txt',
            do_lower_case=True
    ):
        self.vocab_file_path = vocab_file_path
        self.do_lower_case = do_lower_case

        self.text_tokens = None
        self.text_token_labels = None

        self.tokenizer = BertWordPieceTokenizer(vocab_file_path, lowercase=do_lower_case)

    def find_entity_position_in_text(self, entity_tokens):
        text_tokens_length = len(self.text_tokens)
        entity_tokens_length = len(entity_tokens)

        for index in range(text_tokens_length - entity_tokens_length + 1):
            common = [
                entity_token == text_token
                for entity_token, text_token in zip(
                    entity_tokens,
                    self.text_tokens[index: index + entity_tokens_length]
                )
            ]
            if all(common):
                return index

        print(self.text_tokens)
        print(entity_tokens)

        return None

    def labeling_entity(self, entity, entity_type):
        entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False).tokens
        entity_tokens_length = len(entity_tokens)
        index_start = self.find_entity_position_in_text(entity_tokens)

        if index_start is not None:
            self.text_token_labels[index_start] = 'B-' + entity_type
            self.text_token_labels[index_start + 1: index_start + entity_tokens_length] = ['I-' + entity_type] * (entity_tokens_length - 1)

    def labeling_spo_list(self, text, spo_list):
        self.text_tokens = self.tokenizer.encode(text).tokens
        self.text_token_labels = ['O'] * len(self.text_tokens)

        for spo in spo_list:
            subject = spo['subject']
            subject_type = spo['subject_type']

            object_ = spo['object']
            object_type = spo['object_type']

            self.labeling_entity(subject, subject_type)
            self.labeling_entity(object_, object_type)

        for token_index, token in enumerate(self.text_tokens):
            if token.startswith('##'):
                self.text_token_labels[token_index] = '[WP]'

    def separate_raw_data_and_token_labeling(self, file_path, save_path):
        with open(file_path, mode='r', encoding='utf-8') as reader:
            data = json.load(reader)
        reader.close()

        tokenized_data = []

        for paragraph in data:
            text = paragraph['text']
            spo_list = paragraph['spo_list']

            self.labeling_spo_list(text, spo_list)

            tokenized_data.append(paragraph)


if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.separate_raw_data_and_token_labeling('../duie-dataset/train_data_formatted.json', save_path=None)
