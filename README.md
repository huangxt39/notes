some notes:

`transformerlens_bwd_hook.py`: experiments about the question: when changing the activation of a module with forward hooks (say A -> A'), the backward hook will capture the gradient pre- or post- the forward modification (grad for A or A')? The answer is the grad captured by backward hook is the grad for last/newest activation.
