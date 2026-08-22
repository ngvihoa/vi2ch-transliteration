# VietHanBERT

VietHanBERT là mô hình thử nghiệm cho bài toán **Việt → Hán**, được xây dựng dựa trên ý tưởng sử dụng **BERT encoder** của NomBERT nhưng thay phần output head dạng per-token classification bằng **Transformer Decoder** để sinh chuỗi Hán có độ dài khác chuỗi tiếng Việt.

## 1. Bài toán

Input:

```text
Tôi đã học tiếng Hán.
```

Output:

```text
我已學漢。
```

Mô hình nhận một câu tiếng Việt, mã hóa toàn bộ câu bằng BERT Encoder, sau đó sử dụng Transformer Decoder để sinh từng token Hán theo cơ chế autoregressive.

---

## 2. Kiến trúc

```text
Vietnamese sentence
        │
        ▼
Vietnamese Tokenizer
        │
        ▼
Input IDs
        │
        ▼
┌──────────────────────┐
│      BERT Encoder    │
│   (BertModel)        │
└──────────────────────┘
        │
        ▼
Contextual Hidden States
        │
        ▼
┌──────────────────────┐
│ Transformer Decoder  │
│                      │
│ Target Embedding     │
│ + Position Embedding │
│ + Causal Attention   │
│ + Cross Attention    │
└──────────────────────┘
        │
        ▼
Final LayerNorm
        │
        ▼
Linear LM Head
        │
        ▼
Hán Vocabulary
        │
        ▼
Generated Hán sequence
```

### 2.1 Encoder

Model sử dụng:

```python
self.bert = BertModel(
    config,
    add_pooling_layer=False
)
```

BERT Encoder biến chuỗi tiếng Việt thành contextual hidden states:

```text
input_ids
    ↓
BERT
    ↓
H ∈ R^(batch × source_length × hidden_size)
```

Mỗi hidden state chứa thông tin ngữ cảnh của câu tiếng Việt.

### 2.2 Target Embedding

Target Hán có embedding riêng:

```python
self.target_embedding = nn.Embedding(
    config.han_vocab_size,
    config.hidden_size
)
```

Mỗi Hán token được chuyển thành vector có kích thước `hidden_size`.

### 2.3 Target Position Embedding

Decoder sử dụng positional embedding riêng:

```python
self.target_position_embedding = nn.Embedding(
    config.max_target_position_embeddings,
    config.hidden_size
)
```

Target representation được tạo bởi:

```text
target_embedding
+
position_embedding
```

Điều này giúp mô hình phân biệt vị trí của từng token Hán trong câu.

### 2.4 Transformer Decoder

Decoder được xây từ:

```python
nn.TransformerDecoderLayer
```

và:

```python
nn.TransformerDecoder
```

Cấu hình chính:

```text
d_model           = hidden_size
nhead             = decoder_heads
dim_feedforward   = decoder_ffn_dim
num_layers        = decoder_layers
```

Decoder thực hiện hai loại attention chính:

```text
Self-Attention
    ↓
nhìn các token Hán trước đó

Cross-Attention
    ↓
nhìn contextual representation từ BERT Encoder
```

### 2.5 Causal Mask

Decoder sử dụng causal mask:

```python
causal_mask = torch.triu(
    torch.ones(
        target_len,
        target_len,
        device=decoder_input_ids.device,
        dtype=torch.bool
    ),
    diagonal=1
)
```

Mục đích là ngăn decoder nhìn thấy token Hán tương lai.

Ví dụ khi dự đoán:

```text
<BOS> → 我
```

model chưa được phép nhìn thấy:

```text
已 學 漢
```

Khi dự đoán token tiếp theo:

```text
<BOS> 我 → 已
```

model được phép sử dụng:

```text
<BOS> 我
```

nhưng không được nhìn:

```text
學 漢
```

---

## 3. Output Head

Sau decoder:

```python
self.final_layer_norm = nn.LayerNorm(
    config.hidden_size
)
```

Sau đó hidden states được đưa qua linear layer:

```python
self.lm_head = nn.Linear(
    config.hidden_size,
    config.han_vocab_size,
    bias=False
)
```

Nếu:

```text
hidden_size = H
han_vocab_size = V
```

thì mỗi vị trí target sinh ra:

```text
logits ∈ R^V
```

Mỗi logit tương ứng với một token trong Hán vocabulary.

---

## 4. Training Objective

Model sử dụng Cross Entropy Loss:

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=-100
)
```

Các padding label được đặt thành:

```text
-100
```

để không tham gia tính loss.

### Teacher Forcing

Một target:

```text
我 已 學
```

được chuẩn bị thành:

```text
decoder_input_ids:

<BOS> 我 已 學
```

và:

```text
labels:

我 已 學 <EOS>
```

Model học:

```text
<BOS>
  ↓
我

<BOS> 我
  ↓
已

<BOS> 我 已
  ↓
學

<BOS> 我 已 學
  ↓
<EOS>
```

---

## 5. Configuration

VietHanBERT sử dụng configuration riêng:

```python
class VietHanBertConfig(BertConfig):
    model_type = "viet_han_bert"
```

Các tham số thêm cho decoder:

```text
han_vocab_size
decoder_layers
decoder_heads
decoder_ffn_dim
max_target_position_embeddings
```

Ví dụ:

```python
config = VietHanBertConfig(
    vocab_size=...,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=128,
    han_vocab_size=...,
    decoder_layers=4,
    decoder_heads=8,
    decoder_ffn_dim=2048,
    max_target_position_embeddings=128
)
```

---

## 6. Khởi tạo Model

VietHanBERT **không kế thừa `BertPreTrainedModel`** và không load pretrained weights.

Model được xây từ:

```python
class VietHanBertModel(nn.Module):
```

và tất cả `Linear` và `Embedding` được khởi tạo bằng Gaussian initialization:

```python
nn.init.normal_(
    module.weight,
    mean=0.0,
    std=0.02
)
```

Bias của `Linear`:

```text
0
```

LayerNorm:

```text
weight = 1
bias   = 0
```

Do đó phiên bản hiện tại là **training from scratch**.

---

## 7. Padding

Encoder sử dụng:

```text
<PAD> = 0
```

Target decoder cũng sử dụng:

```text
<PAD> = 0
```

Encoder padding được xử lý bằng:

```python
attention_mask
```

Target padding được xử lý bằng:

```python
target_key_padding_mask
```

Trong loss, padding target được đổi thành:

```text
-100
```

để bỏ qua khi tính Cross Entropy.

---

## 8. Forward Pass

Input của model:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=decoder_input_ids,
    labels=labels
)
```

Luồng forward:

```text
input_ids
    │
    ▼
BERT Encoder
    │
    ▼
encoder_hidden_states
    │
    ├───────────────┐
    │               │
    │         Cross Attention
    │               │
    ▼               ▼
decoder_input_ids → Transformer Decoder
                       │
                       ▼
                 LayerNorm
                       │
                       ▼
                    LM Head
                       │
                       ▼
                    logits
                       │
                       ▼
                  Hán tokens
```

Output của model:

```python
{
    "loss": loss,
    "logits": logits,
    "encoder_hidden_states": memory
}
```

Trong đó:

```text
loss
    → training loss

logits
    → xác suất chưa chuẩn hóa trên Hán vocabulary

encoder_hidden_states
    → contextual representation của câu Việt
```

---

## 9. Khác biệt với NomBERT

NomBERT gốc sử dụng hidden state của từng input token để chọn một Hán/Nôm character trong một tập candidate.

VietHanBERT hiện tại thay đổi phần này:

```text
NomBERT:

Vietnamese
    ↓
BERT
    ↓
Per-token classification
    ↓
Hán/Nôm character
```

VietHanBERT:

```text
Vietnamese
    ↓
BERT Encoder
    ↓
Transformer Decoder
    ↓
Autoregressive generation
    ↓
Hán sequence
```

Lý do thay đổi là bài toán Việt → Hán của VietHanBERT dựa trên **parallel sentences**, và source length không nhất thiết bằng target length.

Ví dụ:

```text
Việt Nam
    ↓
越南
```

hoặc:

```text
10 giờ
    ↓
十点
```

Do đó decoder sequence-to-sequence linh hoạt hơn per-token classification.

---

## 10. Vocabulary

VietHanBERT hiện tại sử dụng hai vocabulary riêng:

```text
vocab_vi.json
vocab_han.json
```

Vietnamese vocabulary:

```text
<PAD>
<UNK>
<CLS>
<SEP>
...
```

Hán vocabulary:

```text
<PAD>
<UNK>
<BOS>
<EOS>
...
```

Vocabulary được xây từ training corpus.

`vocab.json` gốc của NomBERT hiện **không phải vocabulary chính của VietHanBERT-v1**. File đó có thể được dùng sau này như một lexical resource để nghiên cứu lexical constraint hoặc candidate bias.

---

## 11. Input / Output Example

### Input

```text
Tôi đã học tiếng Hán.
```

### Vietnamese tokens

```text
[CLS]
Tôi
đã
học
tiếng
Hán
.
[SEP]
```

### Target

```text
<BOS>
我
已
學
漢
。
<EOS>
```

### Model

```text
Vietnamese IDs
    ↓
BERT Encoder
    ↓
Contextual representations
    ↓
Transformer Decoder
    ↓
Hán logits
    ↓
我 已 學 漢 。
```

---

## 12. Current Version

```text
Model name:
VietHanBERT-v1

Task:
Vietnamese → Hán

Encoder:
BERT

Decoder:
Transformer Decoder

Training:
From scratch

Source tokenization:
Vietnamese word/syllable-level + punctuation

Target tokenization:
Hán character-level

Objective:
Autoregressive Cross Entropy

Target vocabulary:
vocab_han.json

Source vocabulary:
vocab_vi.json
```

---

## 13. Future Experiments

Các hướng mở rộng có thể thử sau baseline:

```text
VietHanBERT-v1
    ↓
+ pretrained Vietnamese encoder
    ↓
+ Hán-Việt lexical resource
    ↓
+ lexical candidate bias
    ↓
+ constrained decoding
    ↓
+ reranking
    ↓
+ LLM-based verification
```

Mục tiêu của phiên bản hiện tại là xây dựng một **baseline Việt → Hán hoàn chỉnh**, xác minh pipeline tokenization → encoder → decoder → generation trước khi đưa thêm lexical resource hoặc các cơ chế ràng buộc.
