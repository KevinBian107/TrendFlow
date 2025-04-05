from transformers import BertTokenizer

def load_data():
    from datasets import load_dataset
    dataset = load_dataset("mteb/amazon_massive_scenario", "en")
    # Iterate over each split and print a few examples
    for split, data in dataset.items():
        print(f"--- {split} split ---")
        for i, example in enumerate(data):
            print(example)
            if i >= 4:  # Print only the first 5 examples
                break
    return dataset

def load_tokenizer(args):
    # task1: load bert tokenizer from pretrained "bert-base-uncased", you can also set truncation_side as "left" 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
