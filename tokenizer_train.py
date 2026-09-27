"""
Train a small BPE tokenizer (8 192 tokens, English-only) on FineWeb-Edu.
Run once:  python tokenizer_train.py
The tokenizer is saved to  ./tokenizer/  and reused by all other scripts.
"""
import os
import sys
import json
import itertools
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset

from config import train_cfg

SAVE_DIR = Path(train_cfg.tokenizer_path)
VOCAB_SIZE = train_cfg.tokenizer_vocab_size
SAMPLE_CHARS = 80_000_000   # ~80 MB of text for tokenizer training (fast & sufficient)


def iter_texts(n_chars: int = SAMPLE_CHARS):
    """Stream FineWeb-Edu texts until we have enough characters."""
    ds = load_dataset(
        train_cfg.dataset_name,
        name=train_cfg.dataset_config,
        split=train_cfg.dataset_split,
        streaming=True,
    )
    seen = 0
    for example in ds:
        text: str = example.get("text", "")
        if not text:
            continue
        yield text
        seen += len(text)
        if seen >= n_chars:
            break
    print(f"  [tokenizer] streamed {seen:,} characters for training")


# ChatML special tokens (Qwen / Llama 3.1 style)
# <|im_start|> opens a turn, <|im_end|> closes it.
# <|system|>, <|user|>, <|assistant|> are role marker tokens.
CHAT_SPECIAL_TOKENS = [
    "<|im_start|>",   # start of a chat turn
    "<|im_end|>",     # end of a chat turn
    "<|system|>",     # role: system
    "<|user|>",       # role: user
    "<|assistant|>",  # role: assistant
    "<|tool|>",       # role: tool (for function-calling)
    "<|tool_call|>",  # marks the start of a tool-call JSON block
    "<|endoftext|>",  # GPT-style EOS alias (handy for compatibility)
    "<|pad|>",        # explicit padding alias
]


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Byte-level pre-tokenizer (handles Unicode robustly, keeps spaces)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Base special tokens first (must keep ids 0-3 stable), then chat tokens
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"] + CHAT_SPECIAL_TOKENS

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(iter_texts(), trainer=trainer)

    # Post-processor: prepend <bos> automatically for plain (non-chat) encoding
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A",
        pair="<bos> $A <eos> $B",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    return tokenizer


def main():
    if SAVE_DIR.exists() and (SAVE_DIR / "tokenizer.json").exists():
        print(f"[tokenizer] Already exists at {SAVE_DIR} — skipping training.")
        print("[tokenizer] Delete the directory to retrain.")
        return

    print(f"[tokenizer] Training BPE tokenizer  vocab={VOCAB_SIZE:,} …")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    tok.save(str(SAVE_DIR / "tokenizer.json"))

    # Save config — includes chat special token IDs for fast lookup
    def _id(t): return tok.token_to_id(t)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "model_type": "bpe",
        # Base tokens
        "bos_token":  "<bos>",
        "eos_token":  "<eos>",
        "unk_token":  "<unk>",
        "pad_token":  "<pad>",
        "bos_token_id": _id("<bos>"),
        "eos_token_id": _id("<eos>"),
        "unk_token_id": _id("<unk>"),
        "pad_token_id": _id("<pad>"),
        # ChatML tokens
        "im_start_token":     "<|im_start|>",
        "im_end_token":       "<|im_end|>",
        "system_token":       "<|system|>",
        "user_token":         "<|user|>",
        "assistant_token":    "<|assistant|>",
        "tool_token":         "<|tool|>",
        "tool_call_token":    "<|tool_call|>",
        "endoftext_token":    "<|endoftext|>",
        "im_start_token_id":  _id("<|im_start|>"),
        "im_end_token_id":    _id("<|im_end|>"),
        "system_token_id":    _id("<|system|>"),
        "user_token_id":      _id("<|user|>"),
        "assistant_token_id": _id("<|assistant|>"),
        "tool_token_id":      _id("<|tool|>"),
        "tool_call_token_id": _id("<|tool_call|>"),
        "endoftext_token_id": _id("<|endoftext|>"),
        # Chat template style
        "chat_template": "chatml",
    }
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    actual_vocab = tok.get_vocab_size()
    print(f"[tokenizer] Done. Vocab size: {actual_vocab:,}")
    print(f"[tokenizer] Saved to: {SAVE_DIR.resolve()}")

    # Quick sanity check
    test = "The quick brown fox jumps over the lazy dog."
    enc = tok.encode(test)
    dec = tok.decode(enc.ids)
    print(f"[tokenizer] Sanity  input : {test}")
    print(f"[tokenizer]         tokens: {enc.tokens}")
    print(f"[tokenizer]         decode: {dec}")


if __name__ == "__main__":
    main()
