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

---

## 14. Pipeline dữ liệu thơ

Dataset thơ được tạo theo luồng sau:

```text
Thi Viện
    ↓
crawler theo từng thể loại
    ↓
CSV từng bài trong raw-collections/poetry-collecions/
    ↓
merge thành một CSV cho mỗi thể loại
    ↓
pipelines/poetry_dataset_split/input/
    ↓
chia riêng từng thể loại theo train/test/val
    ↓
poem.train.csv + poem.test.csv + poem.val.csv
    ↓
kaggle-scripts/viet-han-bert.ipynb
```

### 14.1 Định dạng dữ liệu crawl

Mỗi CSV từng bài và CSV đã merge có đúng hai cột:

```csv
vi,ch
Phiên phiên bạch cưu,翩翩白鳩
```

- `vi`: câu phiên âm Hán-Việt.
- `ch`: câu chữ Hán tương ứng.
- Mỗi record phải có đủ cả hai trường.
- Tiêu đề, bản dịch nghĩa, bản dịch thơ và chú thích không được dùng làm cặp
  huấn luyện.

Các crawler hiện có:

| Thể loại | Crawler | Script merge và output theo thể loại |
| --- | --- | --- |
| Kinh Thi | `pipelines/thivien_kinh_thi/scripts/crawl_kinh_thi.py` | `merge_poem_csvs.py` → `outputs/kinhthi.csv` |
| Câu đối | `pipelines/thivien_cau_doi/scripts/crawl_cau_doi.py` | `merge_cau_doi_csvs.py` → `outputs/cau-doi.csv` |
| Ngũ ngôn tứ tuyệt | `pipelines/thivien_ngu_ngon_tu_tuyet/scripts/crawl_ngu_ngon_tu_tuyet.py` | `merge_ngu_ngon_tu_tuyet_csvs.py` → `outputs/ngu-ngon-tu-tuyet.csv` |
| Phú | `pipelines/thivien_phu/scripts/crawl_phu.py` | `merge_phu_csvs.py` → `outputs/phu.csv` |
| Tứ ngôn | `pipelines/thivien_tu_ngon/scripts/crawl_tu_ngon.py` | `merge_tu_ngon_csvs.py` → `outputs/tu-ngon.csv` |
| Thất ngôn cổ phong | `pipelines/thivien_that_ngon_co_phong/scripts/crawl_that_ngon_co_phong.py` | `merge_that_ngon_co_phong_csvs.py` → `outputs/that-ngon-co-phong.csv` |
| Đường luật biến thể | `pipelines/thivien_duong_luat_bien_the/scripts/crawl_duong_luat_bien_the.py` | Chưa có script merge riêng |

Chạy các lệnh từ thư mục gốc `vi2ch-model`. Nên crawl thử một số lượng nhỏ
trước:

```bash
python3 pipelines/thivien_kinh_thi/scripts/crawl_kinh_thi.py \
  --limit 5 --overwrite
```

Sau khi kiểm tra CSV mẫu, dùng `--limit 0` để crawl toàn bộ. Với các crawler
có checkpoint, giữ lại các file progress, URL checkpoint và report trong thư
mục `outputs/` để có thể tiếp tục lần chạy trước. Không giảm thời gian nghỉ
giữa request quá thấp; crawler có cơ chế chờ dài khi Thi Viện trả CAPTCHA.

Sau khi crawl xong, chạy script merge tương ứng. Ví dụ:

```bash
python3 pipelines/thivien_kinh_thi/scripts/merge_poem_csvs.py
python3 pipelines/thivien_ngu_ngon_tu_tuyet/scripts/merge_ngu_ngon_tu_tuyet_csvs.py
python3 pipelines/thivien_tu_ngon/scripts/merge_tu_ngon_csvs.py
```

Script merge kiểm tra header và dữ liệu rỗng trước khi tạo CSV thể loại. Với
pipeline chưa có script merge riêng, cần gom các CSV từng bài thành một CSV
`vi,ch` cho thể loại đó trước khi thực hiện bước chia tập.

### 14.2 Chuẩn bị input để chia tập

Đặt mỗi CSV đã merge vào
`pipelines/poetry_dataset_split/input/`. Mỗi file phải đại diện cho đúng một
thể loại, ví dụ:

```text
pipelines/poetry_dataset_split/input/
├── cau-doi.csv
├── kinhthi.csv
├── ngu-ngon-tu-tuyet.csv
├── phu.csv
└── tu-ngon.csv
```

Không gộp tất cả thể loại thành một file trước bước này. Pipeline dùng ranh
giới file để chia riêng từng thể loại, nhờ đó train, test và validation đều
giữ được dữ liệu từ từng nhóm thơ.

### 14.3 Chia train/test/validation

Chạy:

```bash
python3 pipelines/poetry_dataset_split/scripts/split_poem_csvs.py
```

Cấu hình mặc định:

```text
train = 80%
test  = 10%
val   = 10%
seed  = 42
```

Quy tắc chia:

- Mỗi thể loại được shuffle và chia độc lập trước khi gộp.
- Mỗi input cần ít nhất 3 dòng để cả ba tập đều có dữ liệu của thể loại đó.
- Số dòng được làm tròn nhưng tổng số dòng luôn được bảo toàn.
- Cùng input, tỷ lệ và seed luôn tạo cùng kết quả.
- Seed của từng file phụ thuộc tên file; thêm thể loại mới không làm thay đổi
  cách chia các file đã có.
- Sau khi gộp, từng output được shuffle lại để tránh các block thể loại nằm
  liền nhau.

Output được ghi tại:

```text
pipelines/poetry_dataset_split/outputs/poem.train.csv
pipelines/poetry_dataset_split/outputs/poem.test.csv
pipelines/poetry_dataset_split/outputs/poem.val.csv
```

Có thể đổi tỷ lệ và seed:

```bash
python3 pipelines/poetry_dataset_split/scripts/split_poem_csvs.py \
  --train-ratio 0.7 \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --seed 123
```

Ba tỷ lệ phải lớn hơn `0` và có tổng bằng `1.0`. Có thể dùng
`--input-dir`/`--output-dir` nếu dữ liệu nằm ở vị trí khác. Xem thêm tài liệu
chi tiết tại `pipelines/poetry_dataset_split/README.md`.

### 14.4 Dùng dataset trên Kaggle

Upload ba file output vào Kaggle Dataset. Notebook
`kaggle-scripts/viet-han-bert.ipynb` mặc định tìm các file:

```text
/kaggle/input/datasets/tiennhat/dataset/poem.train.csv
/kaggle/input/datasets/tiennhat/dataset/poem.val.csv
/kaggle/input/datasets/tiennhat/dataset/poem.test.csv
```

Nếu Kaggle mount dataset ở đường dẫn khác, chỉ cần sửa `INPUT_DIR` trong
notebook. Notebook đọc schema `vi,ch`, sau đó đổi tên `ch` thành `cn` trong bộ
nhớ để tương thích với code model hiện tại. Vocabulary chỉ được xây từ
`poem.train.csv`, tránh rò rỉ dữ liệu từ test hoặc validation.

### 14.5 Kiểm tra pipeline chia tập

```bash
python3 -B -m unittest discover \
  -s pipelines/poetry_dataset_split/tests -v
```
