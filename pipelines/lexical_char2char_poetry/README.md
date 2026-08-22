# Lexical char2char vocabulary pipeline

Layer hiện tại chỉ xây vocab ngữ nghĩa `1:N`:

```text
một -> [一, 壹, ...]
```

Các nguồn được xếp theo độ ưu tiên:

1. Mapping đã duyệt trong `resources/curated_vocab.json`.
2. Nghĩa từ vựng độc lập trong `raw-collections/CVDICT.u8`.
3. Cách đọc trong `raw-collections/kVietnamese.json`, nhưng chỉ với ký tự đã tồn tại trong inventory Hán của CVDICT hoặc `hanviet.csv`.
4. Âm Hán–Việt trong `raw-collections/hanviet.csv`.

Pipeline không đọc:

- `raw-collections/chinese-vietnamese.csv`
- `raw-collections/cn-vi/`
- `pipelines/synthetic_poetry/`
- `vocab_old.json`

Chỉ gloss tiếng Việt độc lập gồm một token được đưa vào vocab. Gloss nhiều từ không bị tách máy móc, vì gán mỗi thành phần của một cụm cho cùng một Hán tự sẽ làm sai hợp đồng char2char.

Các glyph Nôm riêng trong `kVietnamese.json` bị loại. `kVietnamese` và `hanviet` chỉ cung cấp bằng chứng cách đọc có ưu tiên thấp hơn mapping đã duyệt và nghĩa CVDICT.

Chạy từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/lexical_char2char_poetry/scripts/build_vocab.py
python3 -B -m unittest discover -s pipelines/lexical_char2char_poetry/tests -v
```

Output trung gian:

- `outputs/vocab.json`: ánh xạ Việt → danh sách Hán tự.
- `outputs/vocab_evidence.jsonl`: nguồn bằng chứng cho từng candidate.
- `outputs/vocab_report.json`: thống kê và danh sách nguồn bị loại.

Layer này chưa đọc thơ, chưa dùng corpus train, chưa chọn một chữ cuối cùng theo ngữ cảnh và chưa tạo dataset.
