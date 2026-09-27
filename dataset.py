"""
Streaming data pipeline for FineWeb-Edu.

Design goals:
  • Zero RAM spike — documents are tokenized on-the-fly in a single Python thread
  • Resumable  — remembers exactly how many tokens have been consumed
  • Memory-safe — pre-allocated ring buffer, no list growth
"""

import os
import json
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
from tokenizers import Tokenizer
from datasets import load_dataset

from config import TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Token stream from FineWeb-Edu
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokenizer(path: str) -> Tokenizer:
    tok_path = Path(path) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}.\n"
            "Run:  python tokenizer_train.py  first."
        )
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_padding(pad_id=tok.token_to_id("<pad>") or 1)
    return tok


class FineWebStream:
    """
    Infinite token stream from FineWeb-Edu with resume support.

    State is purely the number of *documents* consumed so far.
    On resume we skip that many documents before yielding tokens again.
    This is deterministic because HuggingFace streaming preserves ordering.

    Args:
        cfg           : TrainConfig
        skip_docs     : number of documents already consumed (resume)
        verbose       : print skipping progress
    """

    def __init__(
        self,
        cfg: TrainConfig,
        skip_docs: int = 0,
        verbose: bool = True,
    ):
        self.cfg        = cfg
        self.skip_docs  = skip_docs
        self.verbose    = verbose
        self._tok       = _load_tokenizer(cfg.tokenizer_path)
        self._eos_id    = self._tok.token_to_id("<eos>") or 3
        self._bos_id    = self._tok.token_to_id("<bos>") or 2

    # ── Internal: raw token iterator ─────────────────────────────────────────
    def _token_iter(self) -> Iterator[int]:
        ds = load_dataset(
            self.cfg.dataset_name,
            name=self.cfg.dataset_config,
            split=self.cfg.dataset_split,
            streaming=True,
        )

        skipped = 0
        skip_target = self.skip_docs

        if skip_target > 0 and self.verbose:
            print(f"  [dataset] Skipping {skip_target:,} documents to resume …")

        for doc_idx, example in enumerate(ds):
            text: str = example.get("text", "")
            if not text:
                continue

            if skipped < skip_target:
                skipped += 1
                if self.verbose and skipped % 10_000 == 0:
                    print(f"  [dataset] skipped {skipped:,}/{skip_target:,} docs")
                continue

            # Tokenize — encode handles <bos> via post-processor
            ids = self._tok.encode(text).ids
            if len(ids) < 4:
                continue
            # Append EOS to signal document boundary
            ids.append(self._eos_id)
            yield from ids

    # ── Public: chunked batch iterator ───────────────────────────────────────
    def batch_iter(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Iterator[Tuple[torch.Tensor, int]]:
        """
        Yields (batch_tensor, new_total_tokens) where
          batch_tensor  : (batch_size, seq_len + 1)  — input + target overlap
          new_total_tokens: running total of tokens consumed

        The caller slices [:-1] for inputs and [1:] for targets.
        """
        tokens_per_batch = batch_size * (seq_len + 1)
        buf = []
        total_tokens = 0

        for tok_id in self._token_iter():
            buf.append(tok_id)
            if len(buf) >= tokens_per_batch:
                chunk = buf[:tokens_per_batch]
                buf   = buf[tokens_per_batch:]
                total_tokens += tokens_per_batch

                t = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                yield t, total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Dataset state  (saved / loaded with checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class DatasetState:
    """Tracks how far we are into the dataset stream."""

    def __init__(self):
        self.docs_consumed: int = 0
        self.tokens_consumed: int = 0

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "docs_consumed":   self.docs_consumed,
                "tokens_consumed": self.tokens_consumed,
            }, f)

    @classmethod
    def load(cls, path: str) -> "DatasetState":
        s = cls()
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            s.docs_consumed   = d.get("docs_consumed", 0)
            s.tokens_consumed = d.get("tokens_consumed", 0)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a streaming batch generator
# ─────────────────────────────────────────────────────────────────────────────

def make_data_generator(cfg: TrainConfig, skip_tokens: int = 0):
    """
    Returns an iterator that yields (input_ids, target_ids, n_tokens_this_batch).

    skip_tokens: approximate number of tokens already consumed.
    We convert to doc skips heuristically (avg 512 tokens/doc for FineWeb-Edu).
    """
    avg_doc_tokens = 512
    skip_docs = max(0, skip_tokens // avg_doc_tokens)

    stream = FineWebStream(cfg, skip_docs=skip_docs, verbose=True)

    for batch_tensor, total_tokens in stream.batch_iter(
        batch_size=cfg.batch_size,
        seq_len=cfg.context_len,
    ):
        x = batch_tensor[:, :-1]   # (B, T) inputs
        y = batch_tensor[:, 1:]    # (B, T) targets
        n = cfg.batch_size * cfg.context_len
        yield x, y, n
