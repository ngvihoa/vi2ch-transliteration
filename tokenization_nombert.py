import json
import os
from transformers import PreTrainedTokenizer


class NomTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'vocab.json'}
    
    def __init__(
        self,
        vocab_file,
        unk_token='<UNK>',
        unk_token_id=0,
        id_start=1,
        **kwargs
    ):
        self.vocab_file = vocab_file
        self.id_start = id_start
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.pad_token = unk_token
        self.pad_token_id = unk_token_id

        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab_dict = json.load(f)
    
        self.char2id = {}
        self.id2char = {}
        for i, char in enumerate(self.vocab_dict.keys(), start=id_start):
            self.char2id[char] = i
            self.id2char[i] = char
        self.id_to_options = {idx: v for idx, v in enumerate(self.vocab_dict.values(), start=id_start)}

        super().__init__(**kwargs)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.char2id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        if index == self.unk_token_id:
            return self.unk_token
        return self.id2char.get(index, self.unk_token)

    @property
    def vocab_size(self):
        return len(self.char2id) + 1

    def get_vocab(self):
        vocab = {**self.char2id, **self.added_tokens_encoder}
        return vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if filename_prefix:
            vocab_file = os.path.join(save_directory, f'{filename_prefix}-vocab.json')
        else:
            vocab_file = os.path.join(save_directory, 'vocab.json')
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab_dict, f, ensure_ascii=False)
        
        return (vocab_file,)
