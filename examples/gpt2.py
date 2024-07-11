import pequegrad.modules as nn
import pequegrad as pg
import numpy as np
import time
import functools


class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head


class GPT2Attention(nn.StatefulModule):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.split_size = config.n_embd
        self.scale = True

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def _attn(self, q, k, v):
        w = q @ k.transpose(-2, -1)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        w = pg.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        a = pg.matmul(w, v)
        return a

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + [self.n_head, x.size(-1) // self.n_head]
        x = x.reshape(new_x_shape)
        if k:
            return x.permute(1, 0, 2)
        else:
            return x.permute(1, 0, 2)

    def merge_heads(self, x):
        x = x.permute(1, 0, 2)
        new_x_shape = x.size()[:-2] + [
            x.size(-2) * x.size(-1),
        ]
        return x.reshape(new_x_shape)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = (
            x[:, : self.split_size],
            x[:, self.split_size : 2 * self.split_size],
            x[:, -self.split_size :],
        )
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class GPT2MLP(nn.StatefulModule):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = nn.Linear(nx, n_state)
        self.c_proj = nn.Linear(n_state, nx)
        self.act = pg.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class GPT2Block(nn.StatefulModule):
    def __init__(self, config):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=1e-5)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(nx, eps=1e-5)
        self.mlp = GPT2MLP(4 * nx, config)

    def forward(self, x):
        a = self.attn(self.ln_1(x))
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2Model(nn.StatefulModule):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(0.1)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        inputs_embeds = self.wte(input_ids)
        position_ids = pg.arange(
            0, input_shape[-1], dtype=pg.dt.int32, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).reshape((input_shape[-1],))
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.StatefulModule):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits


import time

config = GPT2Config()
model = GPT2LMHeadModel(config).to(pg.device.cuda)
input_ids = pg.Tensor(
    np.random.randint(0, config.vocab_size, (1024,)).astype(np.int32)
).to(pg.device.cuda)
print(input_ids)


@functools.partial(
    pg.jit, externals=model.parameters(), enabled=True
)  # slower than no jit :(
def sample(inputs):
    return model(inputs)


logits = sample(input_ids)
logits = sample(input_ids)
# warmed up

for i in range(10):
    start = time.time()
    logits = sample(input_ids)
    logits.eval()
    print("Time taken: ", time.time() - start)
