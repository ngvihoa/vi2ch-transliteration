# Thi Viện Thất ngôn bát cú shallow crawler

Pipeline tạo một tập test thơ Thất ngôn bát cú từ `PoemType=7`. Đây là crawl
shallow, không cố lấy đủ toàn bộ 5.981 bài.

Phạm vi cố định:

- Chỉ `Country=2`.
- Chỉ các age `52`, `53`, `54`, `55`, `56`, `57`, `2`.
- Partition có tối đa 100 bài được lấy đủ các trang.
- Partition lớn hơn 100 chỉ lấy tối đa 10 trang đầu khi sort `Date` tăng dần
  và 10 trang đầu khi sort `Date` giảm dần.
- Không chia tiếp theo country, age hoặc tác giả.

Với số lượng đã cấu hình, sức chứa lý thuyết của các cửa sổ là tối đa 1.057
URL trước khi loại trùng. Số thực tế có thể thấp hơn do hai cửa sổ thời gian
giao nhau hoặc dữ liệu trên website thay đổi.

Chạy thử 100 bài từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/thivien_that_ngon_bat_cu/scripts/crawl_that_ngon_bat_cu.py \
  --limit 100
```

`--limit` chỉ giới hạn số bài được tải nội dung; URL discovery vẫn hoàn thành
các cửa sổ và lưu checkpoint để lần sau tái sử dụng. Khi có limit, crawler
shuffle URL bằng seed cố định rồi lấy mẫu để tránh chỉ chọn một age đầu tiên;
tăng limit vẫn giữ lại toàn bộ mẫu của lần chạy nhỏ hơn. Dùng `--limit 0` để
tải toàn bộ URL shallow đã tìm được.

Merge các CSV từng bài thành một file `vi,cn`:

```bash
python3 pipelines/thivien_that_ngon_bat_cu/scripts/merge_that_ngon_bat_cu_csvs.py
```

Output merge: `outputs/that-ngon-bat-cu.csv`.
