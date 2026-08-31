# VietHanBERT — Sino-Vietnamese to Han-character Transliteration

VietHanBERT là baseline sequence-to-sequence cho bài toán **âm Hán–Việt → chữ
Hán**. Mô hình dùng **BERT encoder** để mã hóa chuỗi âm Hán–Việt và
**Transformer Decoder** để sinh chuỗi Hán theo cơ chế autoregressive. Decoder
giúp mô hình chọn Hán tự theo ngữ cảnh và mô hình hóa quan hệ giữa các ký tự
đầu ra, thay vì phân loại độc lập từng token nguồn.

- Model đã huấn luyện: [`noah-nguyen-297/VietHanBERT-vi2cn-v1`](https://huggingface.co/noah-nguyen-297/VietHanBERT-vi2cn-v1)
- Notebook huấn luyện: [`kaggle-scripts/vi2cn-transliteration.ipynb`](kaggle-scripts/vi2cn-transliteration.ipynb)
- Script và tài liệu đánh giá: [`evaluate/`](evaluate/README.md)

## 1. Bài toán

Input:

```text
Phiên phiên bạch cưu
```

Output:

```text
翩翩白鳩
```

Mô hình nhận chuỗi phiên âm Hán–Việt, mã hóa toàn bộ chuỗi bằng BERT Encoder,
sau đó sử dụng Transformer Decoder để sinh từng Hán tự theo cơ chế
autoregressive. Đây là bài toán chuyển tự/khôi phục tự dạng, không phải dịch
một câu tiếng Việt hiện đại sang tiếng Hán.

---

## 2. Kiến trúc

```text
Sino-Vietnamese reading
        │
        ▼
Source Tokenizer
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

Mỗi hidden state chứa thông tin ngữ cảnh của chuỗi âm Hán–Việt.

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
Sino-Vietnamese reading
    ↓
BERT Encoder
    ↓
Transformer Decoder
    ↓
Autoregressive generation
    ↓
Hán sequence
```

Với dữ liệu chuẩn, mỗi âm tiết Hán–Việt thường tương ứng với một Hán tự. Ví
dụ `Phiên phiên bạch cưu` có bốn âm tiết và `翩翩白鳩` có bốn Hán tự. Chênh
lệch số token kỹ thuật có thể xuất hiện do token đặc biệt hoặc dấu câu, nhưng
đó không phải lý do chính để xem đây là bài toán sequence-to-sequence.

Lý do dùng decoder là một âm Hán–Việt có thể ứng với nhiều Hán tự. Mô hình cần
ngữ cảnh toàn câu và các ký tự đã sinh để giải quyết nhập nhằng, đồng thời
không buộc mỗi output phải là nhãn độc lập của token nguồn cùng vị trí. Đây là
một lựa chọn kiến trúc cho chuyển tự theo chuỗi; không hàm ý bài toán là dịch
máy hay source/target thường khác số đơn vị nội dung.

---

## 10. Vocabulary

VietHanBERT hiện tại sử dụng hai vocabulary riêng:

```text
vocab_vi.json
vocab_han.json
```

Source Hán–Việt vocabulary:

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
Phiên phiên bạch cưu
```

### Source tokens

```text
[CLS]
Phiên
phiên
bạch
cưu
[SEP]
```

### Target

```text
<BOS>
翩
翩
白
鳩
<EOS>
```

### Model

```text
Source IDs
    ↓
BERT Encoder
    ↓
Contextual representations
    ↓
Transformer Decoder
    ↓
Hán logits
    ↓
翩 翩 白 鳩
```

---

## 12. Phiên bản hiện tại

```text
Model name:
VietHanBERT-vi2cn-v1

Hugging Face:
noah-nguyen-297/VietHanBERT-vi2cn-v1

Task:
Sino-Vietnamese reading → Han characters

Encoder:
BERT

Decoder:
Transformer Decoder

Training:
From scratch

Source tokenization:
Hán–Việt syllable-level + punctuation

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

## 13. Kết quả đánh giá

Baseline được đánh giá ngày **30/08/2026** trên toàn bộ `poem.test.csv` gồm
**2.496** cặp âm Hán–Việt và chữ Hán. Lần chạy dùng greedy decoding trên CPU,
giữ nguyên dấu câu và cố định snapshot Hugging Face tại revision
`10da3a6a9911120cb85da3471337ff001a675245`.

| Metric | Kết quả |
| --- | ---: |
| Exact match / Top-1 ACC | **25,80%** (644/2.496 câu) |
| Character Error Rate (CER) ↓ | **22,54%** |
| Character accuracy (`1 - CER`) ↑ | **77,46%** |
| NEWS mean F-score ↑ | **79,32%** |
| Character BLEU ↑ | **54,94** |
| chrF2 ↑ | **47,22** |
| Teacher-forced token NLL ↓ | **0,9916** |
| Teacher-forced perplexity ↓ | **2,6954** |

Trong 18.026 ký tự reference, Levenshtein breakdown ghi nhận 3.649 phép thay
thế, 354 phép xóa và 60 phép chèn. Khi bỏ dấu câu, exact match tăng lên
**27,20%** (679/2.496 câu); CER là **25,10%** do mẫu số ký tự reference giảm
còn 15.280 và phần lớn lỗi nằm ở Hán tự.

Chạy lại đánh giá từ thư mục gốc:

```bash
python -m pip install -r evaluate/requirements.txt
python evaluate/evaluate_transliteration.py
```

Kết quả đầy đủ và prediction theo từng câu nằm tại
[`evaluate/outputs/metrics.json`](evaluate/outputs/metrics.json) và
[`evaluate/outputs/predictions.csv`](evaluate/outputs/predictions.csv). Xem
[`evaluate/README.md`](evaluate/README.md) để biết định nghĩa metric, smoke
test và các tùy chọn như batch size, device hoặc model revision.

---

## 14. Hướng phát triển

Các hướng mở rộng có thể thử sau baseline:

```text
VietHanBERT-vi2cn-v1
    ↓
+ pretrained Vietnamese/Hán–Việt encoder
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

Mục tiêu của phiên bản hiện tại là xây dựng một **baseline âm Hán–Việt → chữ
Hán hoàn chỉnh**, xác minh pipeline tokenization → encoder → decoder →
generation trước khi đưa thêm lexical resource hoặc các cơ chế ràng buộc.

---

## 15. Pipeline dữ liệu thơ

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
poem.clean.csv + poem.train.csv + poem.test.csv + poem.val.csv
    ↓
kaggle-scripts/vi2cn-transliteration.ipynb
```

### 15.1 Định dạng dữ liệu crawl

Mỗi CSV từng bài và CSV đã merge có đúng hai cột:

```csv
vi,cn
Phiên phiên bạch cưu,翩翩白鳩
```

- `vi`: câu phiên âm Hán-Việt.
- `cn`: câu chữ Hán tương ứng.
- Mỗi record phải có đủ cả hai trường.
- Tiêu đề, bản dịch nghĩa, bản dịch thơ và chú thích không được dùng làm cặp
  huấn luyện.

Các crawler hiện có:

| Thể loại            | Crawler                                                                      | Script merge và output theo thể loại                                    |
| ------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Kinh Thi            | `pipelines/thivien_kinh_thi/scripts/crawl_kinh_thi.py`                       | `merge_poem_csvs.py` → `outputs/kinhthi.csv`                            |
| Câu đối             | `pipelines/thivien_cau_doi/scripts/crawl_cau_doi.py`                         | `merge_cau_doi_csvs.py` → `outputs/cau-doi.csv`                         |
| Ngũ ngôn tứ tuyệt   | `pipelines/thivien_ngu_ngon_tu_tuyet/scripts/crawl_ngu_ngon_tu_tuyet.py`     | `merge_ngu_ngon_tu_tuyet_csvs.py` → `outputs/ngu-ngon-tu-tuyet.csv`     |
| Phú                 | `pipelines/thivien_phu/scripts/crawl_phu.py`                                 | `merge_phu_csvs.py` → `outputs/phu.csv`                                 |
| Tứ ngôn             | `pipelines/thivien_tu_ngon/scripts/crawl_tu_ngon.py`                         | `merge_tu_ngon_csvs.py` → `outputs/tu-ngon.csv`                         |
| Thất ngôn cổ phong  | `pipelines/thivien_that_ngon_co_phong/scripts/crawl_that_ngon_co_phong.py`   | `merge_that_ngon_co_phong_csvs.py` → `outputs/that-ngon-co-phong.csv`   |
| Đường luật biến thể | `pipelines/thivien_duong_luat_bien_the/scripts/crawl_duong_luat_bien_the.py` | `merge_duong_luat_bien_the_csvs.py` → `outputs/duong-luat-bien-the.csv` |
| Thất ngôn bát cú (shallow crawl) | `pipelines/thivien_that_ngon_bat_cu/scripts/crawl_that_ngon_bat_cu.py` | `merge_that_ngon_bat_cu_csvs.py` → `outputs/that-ngon-bat-cu.csv` |

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
`vi,cn` cho thể loại đó trước khi thực hiện bước chia tập.

### 15.2 Chuẩn bị input để chia tập

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

### 15.3 Chia train/test/validation

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
pipelines/poetry_dataset_split/outputs/poem.clean.csv
pipelines/poetry_dataset_split/outputs/poem.train.csv
pipelines/poetry_dataset_split/outputs/poem.test.csv
pipelines/poetry_dataset_split/outputs/poem.val.csv
```

Các artifact hiện tại có 24.943 cặp hợp lệ: 19.954 mẫu train, 2.496 mẫu test
và 2.493 mẫu validation.

`poem.clean.csv` chứa toàn bộ dòng hợp lệ đã gom từ các input, trước khi chia
tập. File này thuận tiện để kiểm tra tổng dữ liệu; quá trình train vẫn sử dụng
ba file split để tránh rò rỉ giữa train, test và validation.

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

### 15.4 Dùng dataset trên Kaggle

Upload ba file split vào Kaggle Dataset. Notebook
`kaggle-scripts/vi2cn-transliteration.ipynb` mặc định tìm các file:

```text
/kaggle/input/datasets/hoanguen/thivien-dataset/poem.train.csv
/kaggle/input/datasets/hoanguen/thivien-dataset/poem.val.csv
/kaggle/input/datasets/hoanguen/thivien-dataset/poem.test.csv
```

Nếu Kaggle mount dataset ở đường dẫn khác, chỉ cần sửa `INPUT_DIR` trong
notebook. Cả notebook và code model đều dùng trực tiếp schema `vi,cn`.
Vocabulary chỉ được xây từ `poem.train.csv`, tránh rò rỉ dữ liệu từ test hoặc
validation.

### 15.5 Kiểm tra pipeline chia tập

```bash
python3 -B -m unittest discover \
  -s pipelines/poetry_dataset_split/tests -v
```
