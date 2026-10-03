# Phrase boosting fixtures

Reference data exported from Python for the Swift `SentencePieceBPEEncoder` and `PhraseBoostTree`.

- `pianissimo-sv-tokenizer.json`: all 8,192 pieces of `KlangAI/pianissimo-sv` `tokenizer.model` as
  `[piece, score, type]` (BPE, `nmt_nfkc`, `add_dummy_prefix`, `split_digits`, extra whitespace kept).
  Klang AI, CC BY 4.0.
- `encode_fixtures.json`: 9,732 texts with the ids and pieces `sentencepiece` produces for them: words,
  names and 200 full sentences from FLEURS `sv_se` (CC BY 4.0) plus edge cases (NFKC ligatures,
  full-width letters, double spaces, digits).
- `tree_fixtures.json`: every state of the boosting tree built by Klang AI's `phrase_boost.py` (a numpy
  port of NeMo's GPU-PB tree) for two phrase lists, `context_score` 1, `depth_scaling` 2. Per state:
  the fallback bonus (full take-back) and every explicit transition except root arcs, which are listed
  once per set under `root` and score `fallback + root score`.
