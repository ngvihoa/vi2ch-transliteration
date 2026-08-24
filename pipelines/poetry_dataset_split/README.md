# Poetry dataset split pipeline

Pipeline nhận nhiều file CSV thơ (mỗi file tương ứng một thể loại), chia riêng
từng thể loại thành `train`/`test`/`val`, sau đó gộp thành ba dataset:

- `outputs/poem.train.csv`
- `outputs/poem.test.csv`
- `outputs/poem.val.csv`

## Input

Đặt các file `*.csv` trực tiếp trong thư mục `input/`. Thư mục này được để
trống trong repository. Mỗi file phải có đúng hai cột như dữ liệu đã crawl:

```csv
vi,ch
Phiên phiên bạch cưu,翩翩白鳩
```

Tên file được dùng để phân biệt thể loại khi chia dữ liệu, nhưng không được
thêm vào output. Các dòng trống hoàn toàn sẽ bị bỏ qua.

## Chạy pipeline

Từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/poetry_dataset_split/scripts/split_poem_csvs.py
```

Mặc định pipeline dùng tỉ lệ `80% train / 10% test / 10% val` và seed `42`.
Mỗi thể loại có ít nhất một dòng trong từng phần (vì vậy mỗi input cần ít nhất
3 dòng). Có thể đổi cấu hình:

```bash
python3 pipelines/poetry_dataset_split/scripts/split_poem_csvs.py \
  --train-ratio 0.7 --test-ratio 0.2 --val-ratio 0.1 --seed 123
```

Có thể truyền thư mục khác bằng `--input-dir` và `--output-dir`. Cùng input,
tỉ lệ và seed sẽ luôn cho cùng kết quả. Thêm một file thể loại mới không làm
thay đổi cách chia các file đã có.

## Test

```bash
python3 -B -m unittest discover -s pipelines/poetry_dataset_split/tests -v
```
