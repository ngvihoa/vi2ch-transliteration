import json
import os

def clean_vocab(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        vocab_old = json.load(f)

    new_vocab = {}
    
    # Chuyển value thành key
    for nom_char, readings in vocab_old.items():
        for reading in readings:
            reading = reading.lower().strip()
            

            if reading and reading not in new_vocab:
                new_vocab[reading] = nom_char

    # Lưu vocab
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_vocab, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(vocab_old)} Nôm characters.")
    print(f"Generated {len(new_vocab)} unique Vietnamese readings in {output_path}.")

if __name__ == "__main__":
    input_file = "vocab_old.json"
    output_file = "vocab.json"
    clean_vocab(input_file, output_file)
