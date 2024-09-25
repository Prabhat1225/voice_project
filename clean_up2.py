
from constant import replacements_hindi
def clean_up(processor,dataset):

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


    def cleanup_text(inputs):
        for src, dst in replacements_hindi:
            inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
        return inputs

    dataset = dataset.map(cleanup_text)

    return dataset,tokenizer

