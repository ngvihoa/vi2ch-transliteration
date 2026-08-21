import string
import re
import json
import os
import unicodedata

def remove_punctuation(text):
    """Loại bỏ tất cả các dấu câu (kể cả tiếng Trung) dựa trên Unicode category"""
    return ''.join(char for char in text if not unicodedata.category(char).startswith('P'))

def clean_vi(text):
    """Làm sạch tiếng Việt: bỏ dấu câu, khoảng trắng thừa, và chuyển chữ thường"""
    text = remove_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def clean_cn(text):
    """Làm sạch tiếng Trung: bỏ dấu câu, khoảng trắng thừa"""
    text = remove_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data():
    pairs = [
        ("dev2021.tok.cn", "dev2021.tok.true.vi"),
        ("train.tok.cn", "train.tok.true.vi"),
        ("tst2021.tok.cn", "tst2021.tok.true.vi")
    ]
    base_dir = os.path.join("raw-collections", "cn-vi")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found")
        return
        
    unique_pairs = set()
    total_lines = 0
    
    for cn_file, vi_file in pairs:
        cn_path = os.path.join(base_dir, cn_file)
        vi_path = os.path.join(base_dir, vi_file)
        
        if not os.path.exists(cn_path) or not os.path.exists(vi_path):
            print(f"Warning: Skipping missing pair ({cn_file}, {vi_file})")
            continue
            
        print(f"Processing: {cn_file} & {vi_file}...")
        
        with open(cn_path, 'r', encoding='utf-8') as f_cn, \
             open(vi_path, 'r', encoding='utf-8') as f_vi:
             
             for cn_line, vi_line in zip(f_cn, f_vi):
                 total_lines += 1
                 cleaned_cn = clean_cn(cn_line)
                 cleaned_vi = clean_vi(vi_line)
                 
                 if cleaned_cn and cleaned_vi:
                     unique_pairs.add((cleaned_vi, cleaned_cn))
                     
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_cleaned.json")
    
    print("Formatting and saving data...")
    data_to_save = [{"vi": vi, "cn": cn} for vi, cn in unique_pairs]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
    print("-" * 30)
    print(f"Total original lines read: {total_lines}")
    print(f"Total unique pairs after cleaning: {len(data_to_save)}")
    print(f"Data successfully saved to: {output_path}")

if __name__ == "__main__":
    process_data()
