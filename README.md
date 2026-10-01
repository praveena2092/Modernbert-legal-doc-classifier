# ModernBERT Legal & Policy Document Classifier

Full fine-tuning of [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) for 30-class classification of long legal and policy documents.

## Task

Given a document's text, predict one of 30 categories. Metric: **accuracy**.

| | |
|---|---|
| Train documents | 50,840 (labeled) |
| Test documents | 12,710 |
| Classes | 30 (imbalanced) |
| Input | Long, variable-length legal / policy text |

## Approach

1. **Cleaning** – collapse whitespace, strip `Page X of Y` artifacts and long dash runs.
2. **Word-level pre-slice** – keep the first 400 and last 1,200 words so tokenization stays cheap on very long documents.
3. **Head + tail truncation** – tokenize and keep the first 128 and last 384 tokens (512 total incl. `[CLS]`/`[SEP]`), since openings and closings of legal documents are the most informative.
4. **Full fine-tuning** with a custom `WeightedTrainer` using class-weighted cross-entropy (weights are inverse-frequency, capped at a 20× max/min ratio).
5. **Stratified 85/15 split** (43,214 train / 7,626 validation), best checkpoint chosen by validation accuracy, early stopping (patience 2).

### Hyperparameters

| Setting | Value |
|---|---|
| Model | ModernBERT-base |
| Max length | 512 (head 128 / tail 384) |
| Learning rate | 2e-5, cosine schedule, 6% warmup |
| Batch size | 4 per device × 4 grad-accum = 16 effective |
| Epochs | 3 |
| Weight decay | 0.01 |
| Precision | fp16 + gradient checkpointing |
| Seed | 42 |

## Results (validation, 7,626 docs)

| Metric | Score |
|---|---|
| Accuracy | **0.7187** |
| Macro F1 | 0.5600 |
| Eval loss | 1.2049 |

The gap between accuracy and macro-F1 reflects the class imbalance — rare categories are harder.

## Repository layout

```
notebooks/modernbert_full_finetune.ipynb   # end-to-end pipeline
data/                                      # put train.csv / test.csv here (git-ignored)
requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
```

Open `notebooks/modernbert_full_finetune.ipynb`, set `train_path` / `test_path` in `CFG`, and run all cells. The notebook has a `debug_mode` flag: run it first with `True` (600 train rows, 1 epoch) to validate the pipeline, then set it to `False` for the full run. A GPU is required for practical training times (the notebook was run on a Kaggle GPU).

Outputs: `submission.csv` (`ID`, `label`), validation/test probability arrays (`.npy`), and the saved model.

## Notes

- Class-probability outputs are saved so this model can be ensembled with others (e.g. the [DeBERTa-LoRA + TF-IDF blend](../deberta-lora-tfidf-legal-doc-classifier)).
- Possible improvements: sliding-window / chunked inference over the full document, longer training, and ensembling with a lexical model.

## License

MIT
