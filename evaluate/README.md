# Evaluate VietHanBERT

Script `evaluate_transliteration.py` đánh giá model Việt → Hán đã publish trên
Hugging Face bằng tập `poem.test.csv`.

## Metrics

- `exact_match_accuracy`: tỷ lệ câu dự đoán khớp hoàn toàn reference, tương
  đương Top-1 ACC trong shared task NEWS về machine transliteration.
- `cer`: Character Error Rate = tổng substitution + deletion + insertion chia
  tổng số ký tự reference. Đây là metric chính; càng thấp càng tốt.
- `character_accuracy`: `1 - CER`, để tiện đọc (có thể âm nếu output lỗi rất
  dài, vì CER không bị chặn ở 1).
- `news_mean_fscore`: F-score dựa trên longest common subsequence của NEWS;
  cho partial credit khi chuỗi chưa khớp hoàn toàn.
- `chrf2`: character n-gram F-score (n-gram 1–6, beta=2).
- `character_bleu`: corpus BLEU với tokenizer ký tự, n-gram 1–4.
- `teacher_forced_token_nll` và `teacher_forced_perplexity`: loss/xác suất của
  model trên target chuẩn; nên dùng để theo dõi checkpoint, không thay thế các
  metric chất lượng chuỗi ở trên.

Metric mặc định giữ dấu câu và bỏ whitespace. Các key có hậu tố `_no_punct`
đánh giá thêm sau khi bỏ dấu câu để tách lỗi nội dung Hán tự khỏi lỗi dấu câu.

## Kết quả baseline

Kết quả chạy ngày 2026-08-30 trên toàn bộ tập test:

| Cấu hình | Giá trị |
| --- | --- |
| Model | `noah-nguyen-297/VietHanBERT-vi2cn-v1` |
| HF revision | `10da3a6a9911120cb85da3471337ff001a675245` |
| Test set | `poem.test.csv` — 2.496 cặp Việt–Hán |
| Decoding | Greedy |
| Device | CPU |
| Max source length | 128 |
| Max new tokens | 127 |
| Thời gian | 194,30 giây |

### Metric chính — giữ dấu câu

| Metric | Kết quả |
| --- | ---: |
| Exact match / Top-1 ACC | **25,80%** — 644/2.496 câu |
| Character Error Rate (CER) | **22,54%** |
| Character accuracy (`1 - CER`) | **77,46%** |
| NEWS mean F-score | **79,32%** |
| Character BLEU | **54,94** |
| chrF2 | **47,22** |
| Teacher-forced token NLL | **0,9916** |
| Teacher-forced perplexity | **2,6954** |

Tổng cộng có 18.026 ký tự reference và 17.732 ký tự prediction. Breakdown
Levenshtein gồm 3.649 substitutions, 354 deletions và 60 insertions.

### Metric phụ — bỏ dấu câu

| Metric | Kết quả |
| --- | ---: |
| Exact match / Top-1 ACC | **27,20%** — 679/2.496 câu |
| Character Error Rate (CER) | **25,10%** |
| Character accuracy (`1 - CER`) | **74,90%** |
| NEWS mean F-score | **76,43%** |
| Character BLEU | **53,39** |
| chrF2 | **45,48** |

CER sau khi bỏ dấu câu cao hơn vì mẫu số chỉ còn 15.280 ký tự, trong khi phần
lớn lỗi nằm ở Hán tự thay vì dấu câu. Exact match vẫn tăng từ 644 lên 679 câu,
cho thấy 35 câu chỉ khác reference ở dấu câu.

Kết quả đầy đủ nằm trong [`outputs/metrics.json`](outputs/metrics.json), còn dự
đoán và thống kê lỗi từng câu nằm trong
[`outputs/predictions.csv`](outputs/predictions.csv).

## Chạy

Từ thư mục `vi2ch-model`:

```bash
python -m pip install -r evaluate/requirements.txt
python evaluate/evaluate_transliteration.py
```

Model và test file mặc định:

```text
model: noah-nguyen-297/VietHanBERT-vi2cn-v1
test : pipelines/poetry_dataset_split/outputs/poem.test.csv
```

Smoke test trước trên 20 mẫu:

```bash
python evaluate/evaluate_transliteration.py --limit 20 --batch-size 4
```

Ví dụ tuỳ chỉnh:

```bash
python evaluate/evaluate_transliteration.py \
  --model-id noah-nguyen-297/VietHanBERT-vi2cn-v1 \
  --test-file pipelines/poetry_dataset_split/outputs/poem.test.csv \
  --batch-size 32 \
  --device cuda
```

Kết quả được lưu tại:

```text
evaluate/outputs/metrics.json
evaluate/outputs/predictions.csv
```

`metrics.json` ghi cả `resolved_revision` (commit SHA thực tế của snapshot HF),
nhờ đó một lần đánh giá có thể được tái lập khi nhánh `main` thay đổi.

Nếu model private, đăng nhập trước bằng `hf auth login` hoặc đặt biến môi
trường `HF_TOKEN`.
