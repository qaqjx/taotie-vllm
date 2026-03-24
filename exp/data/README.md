# Example Data

This directory contains small data assets used by the `blend_kv_v1` examples.

## Files

- `wikimqa_s.jsonl`: example dataset for WikiMQA-style blend evaluation
- `utils/dataset2path.json`: dataset path mapping helper
- `utils/dataset2prompt.json`: prompt template mapping helper
- `utils/dataset2maxlen.json`: dataset length metadata

## Usage

These files are read by evaluation and benchmark utilities such as:

- `blend_wikimqa.py`
- `eval_cache_reuse.py`
- parts of the request-rate benchmark flow

Treat this directory as input data, not generated output.
