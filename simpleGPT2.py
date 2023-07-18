import numpy as np
from utils import load_encoder_hparams_and_params
model_size: str = "124M"
models_dir: str = "models"
encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    attention_scores = softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask)
    return attention_scores @ v

def main(prompt: str, n_tokens_to_generate: int = 10):
    inputs = encoder.encode(prompt)
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]
    
    for _ in range(n_tokens_to_generate):

        x = params['wte'][inputs] + params['wpe'][range(len(inputs))]
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

        for block in params['blocks']:

            # layer norm 1
            ln1 = layer_norm(x, **block['ln_1']) # (seq, 768) => (6, 768)

            # attention
            qkv = linear(ln1, **block['attn']['c_attn']) # => (6, 2304) 

            qkv_heads = np.split(qkv, 3*hparams['n_head'], axis=-1)
            
            attn_out = []
            for head_id in range(0, hparams['n_head']):
                out = attention(qkv_heads[head_id], 
                                qkv_heads[head_id + hparams['n_head']], 
                                qkv_heads[head_id + hparams['n_head']*2], 
                                causal_mask)
                attn_out.append(out)
            
    
            attn = linear(np.hstack(attn_out), **block['attn']['c_proj'])
            
            # residual stream 
            x = x + attn

            # layer norm 2
            ln2 = layer_norm(x, **block['ln_2'])

            # feed forward (or MLP)
            ffn_out = ffn(ln2, **block['mlp'])

            # residual stream 
            x = x + ffn_out
   
        logits =  layer_norm(x[-1], **params['ln_f']) @ params['wte'].T
        
        next_id = np.argmax(logits)
        print(encoder.decode([int(next_id)]), end="", flush=True)
        inputs.append(int(next_id))
    output_ids =  inputs[len(inputs) - n_tokens_to_generate :]  

    output_text = encoder.decode(output_ids)
    #return output_text

if __name__ == "__main__":
    prompt = "not all heroes wear capes"
    main(prompt)
    #print("\n all done! \n output: " + output)
