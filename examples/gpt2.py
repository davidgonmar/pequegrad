"""
Partially from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import os
import sys
import json
from ast import literal_eval
import numpy as np
import pequegrad as pg  # noqa
import pequegrad.modules as pnn  # noqa

# -----------------------------------------------------------------------------


def setup_logging(config):
    """monotonous bookkeeping"""
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))


def assertnotnan(x):
    x = x.numpy() if isinstance(x, pg.Tensor) else x
    assert not np.isnan(x).any(), "found nan in tensor"


class CfgNode:
    """a lightweight configuration class inspired by yacs"""

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(
                obj, leaf_key
            ), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


# -----------------------------------------------------------------------------


class CausalSelfAttention(pnn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = pnn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = pnn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = pnn.Dropout(config.attn_pdrop)
        self.resid_dropout = pnn.Dropout(config.resid_pdrop)
        self.bias = pnn.ModuleParam(
            (
                pg.tril(pg.Tensor.ones((config.block_size, config.block_size))).reshape(
                    (1, config.block_size, config.block_size)
                )
            )
            .astype(pg.dt.float32)
            .eval()
            .to("cuda")
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        T, C = x.size()  # sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=1)

        # compute attention in torch to assert close
        k = k.reshape((T, self.n_head, C // self.n_head)).transpose(
            0, 1
        )  # (B, nh, T, hs)
        q = q.reshape((T, self.n_head, C // self.n_head)).transpose(
            0, 1
        )  # (B, nh, T, hs)
        v = v.reshape((T, self.n_head, C // self.n_head)).transpose(
            0, 1
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_shape = (self.n_head, T, T)
        mask = pg.broadcast_to(self.bias[:, :T, :T], att_shape)
        att = pg.scaled_dot_product_attention(q, k, v, mask, 0.0)
        #  (B, nh, T, hs)
        y = att.transpose(0, 1).reshape(
            (T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class Block(pnn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = pnn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = pnn.LayerNorm(config.n_embd)
        self.mlp = pnn.ModuleDict(
            dict(
                c_fc=pnn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=pnn.Linear(4 * config.n_embd, config.n_embd),
                act=pnn.GELU(),
                dropout=pnn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

        def forward_(x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
            return x

        self.forward_ = forward_

    def forward(self, x):
        return self.forward_(x)


class GPT(pnn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0
        C.resid_pdrop = 0
        C.attn_pdrop = 0
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict(
                {
                    # names follow the huggingface naming conventions
                    # GPT-1
                    "openai-gpt": dict(
                        n_layer=12, n_head=12, n_embd=768
                    ),  # 117M params
                    # GPT-2 configs
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    "gpt2-medium": dict(
                        n_layer=24, n_head=16, n_embd=1024
                    ),  # 350M params
                    "gpt2-large": dict(
                        n_layer=36, n_head=20, n_embd=1280
                    ),  # 774M params
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # (there are a number more...)
                    # I made these tiny models up
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                }[config.model_type]
            )

        self.transformer = pnn.ModuleDict(
            dict(
                wte=pnn.Embedding(config.vocab_size, config.n_embd),
                wpe=pnn.Embedding(config.block_size, config.n_embd),
                drop=pnn.Dropout(config.embd_pdrop),
                h=pnn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=pnn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = pnn.Linear(config.n_embd, config.vocab_size, bias=False)
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        print(
            "number of parameters: %.2fM"
            % (sum(p.numel() for p in self.parameters()) / 1e6)
        )

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]  # ignore these
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla pnn.Linear.
        # this means that we have to transpose these weights when we import them
        # assert len(keys) == len(sd)
        print("n of params in hf model: ", len(keys))
        print("n of params in minGPT model: ", sum([1 for _ in model.parameters()]))
        for k in keys:
            # excepction, lm_head
            if k == "lm_head.weight":
                mm = sd_hf[k].t().contiguous()
                assert list(model.lm_head.weight.shape) == list(
                    mm.shape
                ), f"shape mismatch: {model.lm_head.weight.shape} != {mm.shape} for {k}"
                model.lm_head.weight.assign(pg.Tensor(mm.cpu().numpy()))
                continue
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # need to find sublayers in self and manually assign

                def get_sublayer(key: str):
                    parts = key.split(".")
                    obj = model
                    for p in parts:
                        obj = getattr(obj, p)
                    return obj

                self_w = get_sublayer(k)
                mm = sd_hf[k].contiguous()
                assert list(self_w.shape) == list(
                    mm.shape
                ), f"shape mismatch: {self_w.shape} != {mm.shape} for {k}"
                self_w.assign(pg.Tensor(mm.cpu().numpy()))

            else:
                # vanilla copy over the other parameters

                def get_sublayer(key: str):
                    parts = key.split(".")
                    obj = model
                    for p in parts:
                        obj = getattr(obj, p)
                    return obj

                self_w = get_sublayer(k)
                mm = sd_hf[k]
                assert list(self_w.shape) == list(
                    mm.shape
                ), f"shape mismatch: {self_w.shape} != {mm.shape} for {k}"
                self_w.assign(pg.Tensor(mm.cpu().numpy()))

        return model

    def forward(self, idx):
        (t,) = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe.weight[
            0:t
        ]  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        tokenizer=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.dtype == pg.dt.int32
        self.block_size = 512
        previous_text = tokenizer.decode(idx.numpy())
        print(previous_text, end="", flush=True)
        curr = len(idx) - 1
        # pad idx to max block size
        idx = idx.astype(pg.dt.int32).pad_to(self.block_size).eval().detach().to("cuda")

        @pg.jit.withargs(opts={"common_subexpr_elim": False})
        def runmodel(x, params, curridx, temperature):
            logits = pnn.apply_to_module(self, params, x)
            logits = logits[curridx] / temperature
            probs = pg.softmax(logits, dim=-1)

            return probs.squeeze(0)

        import sys
        import time

        def update_top_line(message):
            # Save the current cursor position
            sys.stdout.write("\033[s")

            # Move cursor to the first line (where the program started), clear it, and print the new message
            sys.stdout.write(
                "\033[1;1H\033[K"
            )  # Move cursor to row 1, column 1 and clear the line
            sys.stdout.write(message + "\n")  # Print the updated message

            # Restore the cursor to its original position
            sys.stdout.write("\033[u")

            sys.stdout.flush()

        di = self.tree_flatten()
        for _ in range(max_new_tokens):
            start = time.time()
            # forward the model to get the logits for the index in the sequence
            probs = runmodel(
                idx,
                di,
                pg.Tensor([curr], device="cuda").astype(pg.dt.int32),
                temperature,
            ).numpy()
            # sample from the distribution
            idx_next = np.random.choice(probs.shape[0], p=probs)
            if curr + 1 < len(idx):
                curr += 1
            else:
                idx = pg.pad_to(idx[1:], self.block_size)
            # append sampled index to the running sequence and continue
            idx = pg.assign_at(
                idx, pg.Tensor(idx_next, device="cuda").astype(pg.dt.int32), curr
            )
            last_token_decoded = tokenizer.decode([idx_next])
            print(last_token_decoded, end="", flush=True)

            # if the last idx is an eos token, we're done
            if last_token_decoded == "<|endoftext|>":
                print("found eos token, stopping")
                break

            update_top_line(f"elapsed time for last token: {time.time() - start:.5f} s")


from transformers import GPT2LMHeadModel, GPT2Tokenizer

use_mingpt = True  # use minGPT or huggingface/transformers model?
model_type = "gpt2"
device = "cuda"

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id  # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval()


def generate(prompt="", num_samples=50, steps=100000, do_sample=True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == "":
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = "<|endoftext|>"
    encoded_input = np.array(tokenizer.encode(prompt))
    encoded_input = (
        pg.Tensor(encoded_input).to("cuda").astype(pg.dt.int32).eval().detach()
    )
    x = encoded_input

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.reshape((-1,))

    # forward the model `steps` times to get samples, in a batch
    model.generate(
        x, max_new_tokens=steps, do_sample=do_sample, top_k=40, tokenizer=tokenizer
    )


generate(
    prompt="How can the net amount of entropy of the universe be massively decreased?",
    num_samples=10,
    steps=10000,
)
"""
from functools import partial
@partial(jit, externals=model.parameters(), enabled=True)
def model_pass(x):
    logits = model(x)
    return logits


# time the forward pass
x = pg.Tensor(np.random.randint(0, 50257, (1024)).astype(np.int32), device="cuda").eval().detach()
print("timing forward pass...")
import time
for _ in range(10):
    start = time.time()
    logits = model_pass(x).eval()
    pg.sync_cuda_device()
    print("elapsed time: %.2f s" % (time.time() - start))

"""
