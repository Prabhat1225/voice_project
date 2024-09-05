# file_1.py
import subprocess

print("Running clean_up2.py")
# Call the next file
subprocess.run(["python", "speaker3.py"])






tokenizer = processor.tokenizer

def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(
    extract_all_chars, 
    batched=True, 
    batch_size=-1, 
    keep_in_memory=True, 
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}

dataset_vocab - tokenizer_vocab

replacements = [
    ('अ', 'a'),
    ('आ', 'aa'),
    ('इ', 'i'),
    ('ई', 'ee'),
    ('उ', 'u'),
    ('ऊ', 'oo'),
    ('ऋ', 'ri'),
    ('ए', 'e'),
    ('ऐ', 'ai'),
    ('ऑ', 'au'),
    ('ओ', 'o'),
    ('औ', 'ou'),
    ('क', 'k'),
    ('ख', 'kh'),
    ('ग', 'g'),
    ('घ', 'gh'),
    ('च', 'ch'),
    ('छ', 'chh'),
    ('ज', 'j'),
    ('झ', 'jh'),
    ('ञ', 'ñ'),
    ('ट', 't'),
    ('ठ', 'th'),
    ('ड', 'd'),
    ('ढ', 'dh'),
    ('ण', 'n'),
    ('त', 't'),
    ('थ', 'th'),
    ('द', 'd'),
    ('ध', 'dh'),
    ('न', 'n'),
    ('प', 'p'),
    ('फ', 'ph'),
    ('ब', 'b'),
    ('भ', 'bh'),
    ('म', 'm'),
    ('य', 'y'),
    ('र', 'r'),
    ('ल', 'l'),
    ('व', 'v'),
    ('श', 'sh'),
    ('ष', 'sh'),
    ('स', 's'),
    ('ह', 'h'),
    ('़', ''),
    ('ा', 'a'),
    ('ि', 'i'),
    ('ी', 'ee'),
    ('ु', 'u'),
    ('ू', 'oo'),
    ('ृ', 'ri'),
    ('ॅ', 'e'),
    ('े', 'e'),
    ('ै', 'ai'),
    ('ॉ', 'o'),
    ('ो', 'o'),
    ('ौ', 'ou'),
    ('्', ''),
    ('क़', 'q'),
    ('ज़', 'z'),
    ('ड़', 'r'),
    ('ढ़', 'rh'),
    ('फ़', 'f'),
    ('।', '.'),
    ('‘', "'"),
    ('’', "'"),
    ('“', '"'),
    ('”', '"'),
    ('ँ', 'n'),
    ('ं', 'm'),
    ('ः', 'h')
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs

dataset = dataset.map(cleanup_text)