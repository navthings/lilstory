from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from datasets import load_dataset


# Load TinyStories
dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train"
)

# BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=[
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[PAD]"
    ]
)

# Train tokenizer
tokenizer.train_from_iterator(
    dataset["text"],
    trainer=trainer
)

# Save it
tokenizer.save("tokenizer.json")

print("Tokenizer saved!")


