some notes:

`transformerlens_bwd_hook.py`: experiments about the question: when changing the activation of a module with forward hooks (say A -> A'), the backward hook will capture the gradient pre- or post- the forward modification (grad for A or A')? The answer is the grad captured by backward hook is the grad for last/newest activation.

`gpt2_tokenizer_nonASCII_char.py`: deals with non ASCII characters when using gpt2 tokenizer.convert_ids_to_tokens(). In some cases we need to keep the tokenized structure (so we cannot use tokenizer.decode()) to associate each token with a value. But these characters will become unrecognizable if one just convert ids into tokens. This file shows a workaround on this.
