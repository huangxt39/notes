from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
import torch
from datasets import load_dataset
import json
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", add_bos_token=True)

raw_str = "Saarbrücken"
input_ids = tokenizer(raw_str)["input_ids"]
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids))
print(tokenizer.decode(input_ids))
print(tokenizer.tokenize(raw_str))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids)))

print([c for c in raw_str])
print([tokenizer.byte_decoder[c] for c in raw_str])
print(bytearray([tokenizer.byte_decoder[c] for c in raw_str]))

test_str = "Søren bor i Kirsebærhaven i København, han har besøgt Saarbrücken siste år."
test_str = "Bien sûr, voici un texte aléatoire en français: et à trouver le trésor qui"
test_str = "ü ÿ å û ß Å Æ bü bÿ bå bû bß bÅ bÆ"

# method 1
tokens = tokenizer.convert_ids_to_tokens(tokenizer(test_str)["input_ids"])
print(tokens)
print("".join([t.replace("Ġ", " ") for t in tokens]))

# method 2
new_decoding = [bytearray([tokenizer.byte_decoder[c] for c in t]).decode("utf-8", errors="replace") for t in tokens]
print(new_decoding)
print("".join(new_decoding))

# method 3
def rearrange(tokens):
    # this changes the model prediction, it's just a temporary workaround
    tokens = tokens.copy()
    for i in range(len(tokens)):
        if tokens[i] == "ĠÃ" and (i < len(tokens)-1):
            tokens[i] = "Ġ_"
            tokens[i+1] = "Ã" + tokens[i+1]
        elif tokens[i] == "Ã" and (i < len(tokens)-1):
            tokens[i] = "_"
            tokens[i+1] = "Ã" + tokens[i+1]
        
    return tokens

tokens = rearrange(tokens)
new_decoding = [bytearray([tokenizer.byte_decoder[c] for c in t]).decode("utf-8", errors="replace") for t in tokens]
print(new_decoding)
print("".join(new_decoding))

