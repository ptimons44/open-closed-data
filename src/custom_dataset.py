import os
import datasets

def get_custom_dataset(dataset_config, tokenizer, split="train"):
    assert split == "train", f"not capable of using split \"{split}\""
    assert tokenizer.pad_token == "[PAD]", "TODO: is this what we want for padding token??"
    assert tokenizer.padding_size == "right", "TODO: is this what we want for padding side??"
    dataset = datasets.load_dataset(
        "DataProvenanceInitiative/Commercially-Verified-Licenses",
        split=split,
        num_proc=os.cpu_count(),
        revision="main",
        data_files="data/dolly_15k/*.jsonl"
    ).select(range(100)) # TODO: remove this subset for debugging purposes

    tokenized = [tokenizer.batch_encode_plus((example["inputs"], example["labels"]), truncation=True, padding='longest') for example in dataset]
    input_ids, attention_mask, labels = [], [], []
    for example in tokenized:
        idx = len(example["input_ids"]) // 2
        input_ids.append(example["input_ids"][0])
        labels.append(example["input_ids"][1])
        attention_mask.append(example["attention_mask"][0])

    new_dataset = datasets.Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    })

    return new_dataset
