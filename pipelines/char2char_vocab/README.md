# Lexical char2char vocabulary pipeline

Layer hiện tại chỉ xây vocab ngữ nghĩa `1:N`:

```text
một -> [一, 壹, ...]
không có ánh xạ -> [𠀗]
```

Các nguồn được xếp theo độ ưu tiên:

1. Mapping đã duyệt trong `resources/curated_vocab.json`.
2. Nghĩa từ vựng độc lập trong `raw-collections/CVDICT.u8`.
3. Cách đọc trong `raw-collections/kVietnamese.json` và `raw-collections/hanviet.csv` chỉ dùng để nhận diện/lọc candidate thuần âm; chúng không tự sinh candidate mới và không tác động thứ hạng nghĩa.

Pipeline không đọc:

- `raw-collections/chinese-vietnamese.csv`
- `raw-collections/cn-vi/`
- `vocab_old.json`

Chỉ gloss tiếng Việt độc lập gồm một token được đưa vào vocab. Gloss nhiều từ không bị tách máy móc, vì gán mỗi thành phần của một cụm cho cùng một Hán tự sẽ làm sai hợp đồng char2char.

Các glyph Nôm riêng trong `kVietnamese.json` bị loại. Candidate chỉ có bằng chứng âm đọc cũng bị lọc; token không còn candidate hợp lệ nhận placeholder `𠀗`.

Chạy từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/char2char_vocab/scripts/build_vocab.py
python3 -B -m unittest discover -s pipelines/char2char_vocab/tests -v
```

Output trung gian:

- `outputs/vocab.json`: ánh xạ Việt → danh sách Hán tự.
- `outputs/vocab_evidence.jsonl`: nguồn bằng chứng cho từng candidate.
- `outputs/vocab_report.json`: thống kê và danh sách nguồn bị loại.

Pipeline này không đọc thơ và không tạo dữ liệu thơ. Việc sử dụng vocab để tạo data thuộc pipeline độc lập `pipelines/poetry_char2char_data/`.
