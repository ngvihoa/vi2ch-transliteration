# Thi Vien Kinh Thi crawler

Pipeline này thu thập các cặp phiên âm Hán-Việt và nguyên tác chữ Hán từ danh
sách Kinh Thi trên Thi Viện. Phần dịch nghĩa và các bản dịch thơ không được lấy.

Mỗi bài tạo một file CSV riêng trong `raw-collections/poetry-collecions/`:

```text
ankyloi1.csv
```

CSV có đúng hai cột `vi,cn`. Mỗi record là một câu phiên âm Hán-Việt và câu
chữ Hán tương ứng. Tiêu đề, phần dịch nghĩa và chú thích không được đưa vào
CSV. Chữ Hán được giữ theo dạng xuất hiện trong nguyên tác trên Thi Viện.

## Chạy thử 5 bài

Từ thư mục gốc `vi2ch-model`:

```bash
python3 pipelines/thivien_kinh_thi/scripts/crawl_kinh_thi.py --limit 5 --overwrite
python3 -B -m unittest discover -s pipelines/thivien_kinh_thi/tests -v
```

`--limit` mặc định là `5`. Crawler nghỉ một giây giữa các request; có thể đổi
bằng `--delay`, nhưng nên giữ tốc độ thấp để không tạo tải không cần thiết cho
website nguồn.

## Chạy toàn bộ danh sách

Sau khi kiểm tra dữ liệu mẫu:

```bash
python3 pipelines/thivien_kinh_thi/scripts/crawl_kinh_thi.py --limit 0 --overwrite
```

`--limit 0` nghĩa là không giới hạn. Nếu có hai bài tạo ra cùng tên file,
crawler thêm UID của Thi Viện vào tên bài bị trùng để tránh ghi đè.

Crawler lấy danh sách qua cây nhóm `Quốc phong`, `Nhã`, `Tụng` để tránh giới
hạn 100 kết quả của trang tìm kiếm. Bài chỉ có nhan đề hoặc không có cặp nội
dung hợp lệ sẽ được bỏ qua và ghi tại `outputs/crawl_report.json`.

Nguồn: <https://www.thivien.net/Khổng-Tử/Thi-kinh-Kinh-thi/group-ZDB2Tl5514uy8PI478SU_g>
