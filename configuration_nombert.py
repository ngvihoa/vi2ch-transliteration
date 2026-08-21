from transformers import BertConfig


class NomBertConfig(BertConfig):
    def __init__(self, unk_id=0, id_start=1, output_vocab_size=7430, lm_head_dict={}, **kwargs):
        super().__init__(**kwargs)
        self.unk_id = unk_id
        self.id_start = id_start
        self.output_vocab_size = output_vocab_size
        self.lm_head_dict = lm_head_dict
