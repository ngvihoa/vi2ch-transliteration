# Poetry char2char data pipeline

Pipeline này tạo dữ liệu char2char từ thơ và hoàn toàn tách khỏi bước build vocab.

Input:

- Thơ Quốc ngữ nguồn: `raw-collections/poem/*.vi.txt`.
- Vocab đã build: `pipelines/char2char_vocab/outputs/vocab.json`.

Pipeline chỉ đọc hai input trên; nó không sửa hoặc build lại vocab và không dùng corpus train.
Các file `*.no.txt` là target Nôm có sẵn và không được đưa vào nhánh tự sinh này.

Quy tắc ánh xạ:

- Mỗi token Việt chọn candidate hạng đầu trong vocab.
- Token OOV hoặc vocab placeholder nhận `𠀗`.
- `target_chars` luôn có đúng số phần tử bằng `source_tokens`.

Chạy từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/poetry_char2char_data/scripts/map_raw_poems.py
python3 -B -m unittest discover -s pipelines/poetry_char2char_data/tests -v
```

Output trung gian:

- `outputs/poem_char2char.jsonl`: toàn bộ dòng thơ đã ánh xạ.
- `outputs/poem_char2char_review.jsonl`: các dòng chứa placeholder.
- `outputs/poem_char2char_report.json`: độ phủ và kiểm tra alignment.

Các file này chưa phải dataset cuối đã kiểm định.
