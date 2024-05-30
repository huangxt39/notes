from transformer_lens import HookedTransformer
import torch

torch.set_printoptions(sci_mode=False)
hooked_model = HookedTransformer.from_pretrained("gpt2")
tokenizer = hooked_model.tokenizer

grad_dict = {}
def fwd_hook(tensor, hook):
    tensor.register_hook(lambda grad: grad_dict.update({"torch_hook_pre": grad}))
    tensor = tensor * 2
    tensor.register_hook(lambda grad: grad_dict.update({"torch_hook_post": grad}))
    return tensor

def fwd_hook2(tensor, hook):
    tensor = tensor * 2
    tensor.register_hook(lambda grad: grad_dict.update({"torch_hook_post2": grad}))
    return tensor

def bwd_hook(tensor, hook):
    # grad for the output of the module
    grad_dict.update({"lens_hook": tensor})

hooked_model.hook_dict["blocks.6.hook_resid_mid"].add_hook(fwd_hook, dir="fwd")
hooked_model.hook_dict["blocks.6.hook_resid_mid"].add_hook(fwd_hook2, dir="fwd")
hooked_model.hook_dict["blocks.6.hook_resid_mid"].add_hook(bwd_hook, dir="bwd")

batch_tokens = hooked_model.to_tokens(["hello, how are you?"])

logits = hooked_model(batch_tokens, return_type="logits") * 10000
logits.sum().backward()

print((grad_dict["torch_hook_pre"] == grad_dict["lens_hook"]).all())
print((grad_dict["torch_hook_post"] == grad_dict["lens_hook"]).all())
print((grad_dict["torch_hook_post2"] == grad_dict["lens_hook"]).all())
print((grad_dict["torch_hook_pre"] == grad_dict["torch_hook_post"]).all())
print((grad_dict["torch_hook_post"] == grad_dict["torch_hook_post2"]).all())
print((grad_dict["torch_hook_pre"] == grad_dict["torch_hook_post2"]).all())

print(grad_dict["torch_hook_pre"][0, -1, :5])
print(grad_dict["torch_hook_post"][0, -1, :5])
print(grad_dict["torch_hook_post2"][0, -1, :5])
print(grad_dict["lens_hook"][0, -1, :5])

# output:
# Loaded pretrained model gpt2 into HookedTransformer
# tensor(False, device='mps:0')
# tensor(False, device='mps:0')
# tensor(True, device='mps:0')
# tensor(False, device='mps:0')
# tensor(False, device='mps:0')
# tensor(False, device='mps:0')
# tensor([ 0.8330,  0.9233, -0.9324,  0.6335,  3.6556], device='mps:0')
# tensor([ 0.4165,  0.4616, -0.4662,  0.3168,  1.8278], device='mps:0')
# tensor([ 0.2083,  0.2308, -0.2331,  0.1584,  0.9139], device='mps:0')
# tensor([ 0.2083,  0.2308, -0.2331,  0.1584,  0.9139], device='mps:0')
