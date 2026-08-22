import json
import os
import re
import unicodedata


BASE_DIR = os.path.join("raw-collections", "cn-vi")
OUTPUT_DIR = "dataset"


SPLITS = {
    "train": ("train.tok.cn", "train.tok.true.vi"),
    "dev": ("dev2021.tok.cn", "dev2021.tok.true.vi"),
    "test": ("tst2021.tok.cn", "tst2021.tok.true.vi"),
}


def normalize_text(text):
    """
    Chuẩn hóa text nhưng KHÔNG:
    - xóa dấu câu
    - chuyển lowercase

    Vì đây là bài toán Việt -> Hán,
    punctuation và casing có thể mang thông tin.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_vi(text):
    """Clean Vietnamese sentence."""
    return normalize_text(text)


def clean_cn(text):
    """Clean Chinese/Hán sentence."""
    return normalize_text(text)


def read_parallel_files(cn_path, vi_path):
    """
    Đọc hai file parallel và kiểm tra alignment.
    """

    with open(cn_path, "r", encoding="utf-8") as f_cn:
        cn_lines = f_cn.readlines()

    with open(vi_path, "r", encoding="utf-8") as f_vi:
        vi_lines = f_vi.readlines()

    if len(cn_lines) != len(vi_lines):
        raise ValueError(
            f"\nLINE COUNT MISMATCH!\n"
            f"Chinese file: {len(cn_lines):,} lines\n"
            f"Vietnamese file: {len(vi_lines):,} lines\n"
            f"CN: {cn_path}\n"
            f"VI: {vi_path}\n"
        )

    return cn_lines, vi_lines


def process_split(split_name, cn_file, vi_file):
    """
    Process một split: train / dev / test.
    """

    cn_path = os.path.join(BASE_DIR, cn_file)
    vi_path = os.path.join(BASE_DIR, vi_file)

    if not os.path.exists(cn_path):
        print(f"[ERROR] Missing file: {cn_path}")
        return None

    if not os.path.exists(vi_path):
        print(f"[ERROR] Missing file: {vi_path}")
        return None

    print("\n" + "=" * 60)
    print(f"Processing split: {split_name.upper()}")
    print("=" * 60)

    cn_lines, vi_lines = read_parallel_files(cn_path, vi_path)

    total_lines = len(cn_lines)

    unique_pairs = set()

    empty_pairs = 0
    duplicate_pairs = 0

    for cn_line, vi_line in zip(cn_lines, vi_lines):

        cleaned_cn = clean_cn(cn_line)
        cleaned_vi = clean_vi(vi_line)

        # Bỏ cặp rỗng
        if not cleaned_cn or not cleaned_vi:
            empty_pairs += 1
            continue

        pair = (cleaned_vi, cleaned_cn)

        if pair in unique_pairs:
            duplicate_pairs += 1
            continue

        unique_pairs.add(pair)

    # Sort để output deterministic
    data = [
        {
            "vi": vi,
            "cn": cn
        }
        for vi, cn in sorted(unique_pairs)
    ]

    print(f"Original pairs : {total_lines:,}")
    print(f"Empty pairs    : {empty_pairs:,}")
    print(f"Duplicates     : {duplicate_pairs:,}")
    print(f"Final pairs    : {len(data):,}")

    return data


def save_json(data, filename):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved to: {output_path}")


def process_data():

    print("=" * 60)
    print("Vietnamese -> Han Parallel Corpus Preprocessing")
    print("=" * 60)

    total_pairs = 0

    for split_name, (cn_file, vi_file) in SPLITS.items():

        data = process_split(
            split_name,
            cn_file,
            vi_file
        )

        if data is None:
            continue

        output_filename = f"{split_name}.json"

        save_json(
            data,
            output_filename
        )

        total_pairs += len(data)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Total final pairs: {total_pairs:,}")

    print("\nOutput files:")

    for split_name in SPLITS:
        print(
            f"  {split_name}: "
            f"{os.path.join(OUTPUT_DIR, split_name + '.json')}"
        )


if __name__ == "__main__":
    process_data()