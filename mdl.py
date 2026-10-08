import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerFast


# device

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("using:", device)


# settings

steps = int(input("steps: "))

block_size = 384
batch_size = 32


# dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:10%]"
)

print(dataset)


# tokenizer

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    unk_token="[UNK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    pad_token="[PAD]"
)

vocab_size = tokenizer.vocab_size


def encode(s):
    return tokenizer.encode(s)


def decode(ids):
    return tokenizer.decode(ids)


# encode dataset

text = "\n".join(dataset["text"])

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print("tokens:", len(data))
print("vocab:", vocab_size)


# train / validation split

split = int(len(data) * 0.9)

train_data = data[:split]
val_data = data[split:]


# training batches

def get_batch():

    starts = torch.randint(
        0,
        len(train_data) - block_size - 1,
        (batch_size,)
    )

    offsets = torch.arange(block_size)

    x = train_data[
        starts[:, None] + offsets
    ]

    y = train_data[
        starts[:, None] + offsets + 1
    ]

    return x.to(device), y.to(device)


# validation batches

def get_val_batch():

    starts = torch.randint(
        0,
        len(val_data) - block_size - 1,
        (batch_size,)
    )

    offsets = torch.arange(block_size)

    x = val_data[
        starts[:, None] + offsets
    ]

    y = val_data[
        starts[:, None] + offsets + 1
    ]

    return x.to(device), y.to(device)


# model

config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=block_size,
    rms_norm_eps=1e-5,
    rope_theta=10000,
    tie_word_embeddings=False
)

model = LlamaForCausalLM(config).to(device)

print(
    "parameters:",
    sum(p.numel() for p in model.parameters())
)


# optimizer

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# validation loss

@torch.no_grad()
def validation_loss():

    model.eval()

    x, y = get_val_batch()

    logits = model(x).logits

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    model.train()

    return loss.item()


# train

model.train()

for step in range(steps):

    x, y = get_batch()

    logits = model(x).logits

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # print training + validation loss

    if step % 100 == 0:

        val_loss = validation_loss()

        print(
            f"step {step} | "
            f"train loss {loss.item():.3f} | "
            f"val loss {val_loss:.3f}"
        )


# generation

@torch.no_grad()
def generate(
    prompt,
    length=500,
    temperature=0.8
):

    model.eval()

    tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    for _ in range(length):

        context = tokens[:, -block_size:]

        logits = model(context).logits
        logits = logits[:, -1, :]

        logits = logits / temperature

        probs = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            1
        )

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

    return decode(tokens[0].tolist())


# save

model.save_pretrained(
    "lilstoryteller"
)

tokenizer.save_pretrained(
    "lilstoryteller"
)

print("\nmodel saved")


# output

print("\nmodel output\n")

print(
    generate(
        "Once upon a time ",
        500,
        temperature=0.8
    )
)
