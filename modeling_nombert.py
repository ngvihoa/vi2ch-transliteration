import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from .configuration_nombert import NomBertConfig


class NomBertModel(BertPreTrainedModel):
    config_class = NomBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.max_position_embeddings = config.max_position_embeddings
        self.lm_head_dict = config.lm_head_dict
        self.registered_token_ids = list(map(int, config.lm_head_dict.keys()))
        self.lm_head = nn.Embedding(config.output_vocab_size, config.hidden_size)

    def forward(self, input_ids, labels=None, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        registered_token_ids_tensor = torch.tensor(
            self.registered_token_ids, 
            device=input_ids.device
        )
        valid_token_mask = torch.isin(input_ids, registered_token_ids_tensor)
        valid_mask = valid_token_mask & attention_mask.bool()

        loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        for token_id_str in self.lm_head_dict.keys():
            token_id = int(token_id_str)
            mask = (input_ids == token_id) & valid_mask
            selected_hidden = hidden_states[mask]
            selected_labels = labels[mask] if labels is not None else None
            
            if selected_hidden.size(0) == 0:
                continue

            lm_head_ids = self.lm_head_dict[token_id_str]
            lm_head_ids_tensor = torch.tensor(lm_head_ids, device=input_ids.device)
            lm_head = self.lm_head(lm_head_ids_tensor)
            logits = torch.matmul(selected_hidden, lm_head.T)
            
            if labels is not None:
                loss = loss + F.cross_entropy(
                    logits, 
                    selected_labels, 
                    ignore_index=-100
                )
        
        return {'loss': loss} if labels is not None else outputs

    def parse_nom_text(self, tokenizer, texts, post_normalize=True, batch_size=None):
        max_length = self.max_position_embeddings
        segments_info = []
        for text_idx, text in enumerate(texts):
            segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            for seg_idx, seg in enumerate(segments):
                segments_info.append((text_idx, seg_idx, seg))
        
        all_segments = [seg for _, _, seg in segments_info]
        all_pred_chars = []
        all_pred_probs = []

        if batch_size is None:
            batch_size = len(texts)
        for i in range(0, len(all_segments), batch_size):
            batch_segments = all_segments[i:i+batch_size]
            batch_pred_chars, batch_pred_probs = self._parse_nom_text_batch(tokenizer, batch_segments)
            all_pred_chars.extend(batch_pred_chars)
            all_pred_probs.extend(batch_pred_probs)

        text_results = {}
        for text_idx in range(len(texts)):
            text_results[text_idx] = {'chars': [], 'probs': []}
        
        for (text_idx, seg_idx, _), pred_chars, pred_probs in zip(segments_info, all_pred_chars, all_pred_probs):
            text_results[text_idx]['chars'].append((seg_idx, pred_chars))
            text_results[text_idx]['probs'].append((seg_idx, pred_probs))

        output_texts = []
        all_outputs_probs = []
        for text_idx in range(len(texts)):
            sorted_chars = sorted(text_results[text_idx]['chars'], key=lambda x: x[0])
            sorted_probs = sorted(text_results[text_idx]['probs'], key=lambda x: x[0])
            
            merged_chars = []
            merged_probs = []
            for seg_idx, chars in sorted_chars:
                merged_chars.extend(chars)
            for seg_idx, probs in sorted_probs:
                merged_probs.extend(probs)

            output_text = ''
            for i, (char, processed) in enumerate(merged_chars):
                output_text += char
                if i < len(merged_chars)-1 and (processed or merged_chars[i+1][1]):
                    output_text += ' '

            if post_normalize:
                output_text = self.post_normalize(output_text)
            output_texts.append(output_text)
            all_outputs_probs.append(merged_probs)

        return output_texts, all_outputs_probs

    def _parse_nom_text_batch(self, tokenizer, segments):
        encoded = tokenizer.batch_encode_plus(
            segments,
            add_special_tokens=False,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_position_embeddings
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        batch_size = len(segments)
        id_to_options_ids = list(tokenizer.id_to_options.keys())
        id_to_options_tensor = torch.tensor(id_to_options_ids, device=self.device)
        registered_ids = torch.tensor(self.registered_token_ids, device=self.device)
        valid_mask = (
            torch.isin(input_ids, registered_ids) & 
            attention_mask.bool()
        )

        pred_chars = [[(c, False) for c in seg] for seg in segments]
        pred_probs = [[] for _ in range(batch_size)]

        if valid_mask.any():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            batch_indices, seq_indices = torch.where(valid_mask)
            token_ids = input_ids[batch_indices, seq_indices]
            hidden_vecs = hidden_states[batch_indices, seq_indices]

            for token_id_str in self.lm_head_dict:
                token_id = int(token_id_str)
                token_mask = (token_ids == token_id)
                if not token_mask.any():
                    continue

                token_hidden = hidden_vecs[token_mask]
                token_batch = batch_indices[token_mask]
                token_seq = seq_indices[token_mask]

                lm_head_ids = self.lm_head_dict[token_id_str]
                lm_head_ids_tensor = torch.tensor(lm_head_ids, device=input_ids.device)
                lm_head = self.lm_head(lm_head_ids_tensor)
                logits = torch.matmul(token_hidden, lm_head.T)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                for i, (b, s) in enumerate(zip(token_batch.tolist(), token_seq.tolist())):
                    options = tokenizer.id_to_options[token_id]
                    char = options[preds[i].item()]

                    pred_chars[b][s] = (char, True)

                    candidates = sorted(
                        [(opt, probs[i][j].item()) for j, opt in enumerate(options)],
                        key=lambda x: x[1], reverse=True
                    )

                    if s >= len(pred_probs[b]):
                        pred_probs[b].extend([{}] * (s - len(pred_probs[b]) + 1))

                    pred_probs[b][s] = {
                        'char': segments[b][s],
                        'candidates': candidates
                    }

        single_option_mask = (
            attention_mask.bool() &
            torch.isin(input_ids, id_to_options_tensor) &
            ~torch.isin(input_ids, registered_ids)
        )
        batch_indices_single, seq_indices_single = torch.where(single_option_mask)

        for b, s in zip(batch_indices_single.tolist(), seq_indices_single.tolist()):
            token_id = input_ids[b, s].item()
            options = tokenizer.id_to_options[token_id]

            pred_chars[b][s] = (options[0], True)

            if s >= len(pred_probs[b]):
                pred_probs[b].extend([{}] * (s - len(pred_probs[b]) + 1))

            pred_probs[b][s] = {
                'char': segments[b][s],
                'candidates': [(options[0], 1.0)]
            }

        for b in range(batch_size):
            seg_len = len(segments[b])
            pred_chars[b] = pred_chars[b][:seg_len]
            for s in range(seg_len):
                if s < len(pred_probs[b]) and pred_probs[b][s]:
                    continue
                char = segments[b][s]
                if s >= input_ids.shape[1]:
                    token_id = 0
                else:
                    token_id = input_ids[b, s].item()

                candidates = [(char, 1.0)]
                if token_id != 0 and token_id in tokenizer.id_to_options:
                    options = tokenizer.id_to_options[token_id]
                    if len(options) == 1:
                        candidates = [(options[0], 1.0)]

                if s >= len(pred_probs[b]):
                    pred_probs[b].extend([{}] * (s - len(pred_probs[b]) + 1))

            pred_probs[b] = pred_probs[b][:seg_len]

        pred_probs = [[p for p in batch if p != {}] for batch in pred_probs]
        
        return pred_chars, pred_probs

    def post_normalize(self, text):
        text = re.sub(r'\s*[。\.]', '.', text)
        text = re.sub(r'\s*[，、,]', ',', text)
        text = re.sub(r'\s*[！!]', '!', text)
        text = re.sub(r'\s*[？\?]', '?', text)
        return text
