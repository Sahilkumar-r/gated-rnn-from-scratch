# Deep Gated RNN — Residual Bidirectional GRU for Text Classification

A deep, gated recurrent network built in PyTorch and trained from scratch on the
[IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
dataset from Kaggle (binary sentiment classification).

## Architecture

```
tokens ─► Embedding ─► Dropout ─► Linear (project to 2·H)
        ─► [ BiGRU ─► Dropout ─► + residual ─► LayerNorm ] × N
        ─► Masked attention pooling ⊕ Masked max pooling
        ─► MLP head ─► sentiment logit
```

- **Gating:** GRU update/reset gates in every layer (4 stacked blocks by default).
- **Depth made trainable:** residual connections + LayerNorm around every recurrent block.
- **Padding-aware:** packed sequences in the RNN, masks in the pooling layers.
- **Training:** AdamW, OneCycle LR, gradient clipping, mixed precision, early stopping.

Default config: 256-d embeddings, 256 hidden units per direction, 4 layers, 30k vocab, 400 max tokens.

## Quickstart

```bash
git clone https://github.com/<your-username>/deep-gated-rnn.git
cd deep-gated-rnn
pip install -r requirements.txt
jupyter notebook deep_gated_rnn.ipynb
```

### Kaggle credentials
`kagglehub` downloads the dataset automatically. Outside Kaggle, authenticate with one of:
- `kagglehub.login()` inside the notebook, or
- an API token at `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token).

The notebook runs on Google Colab, Kaggle Notebooks, or any machine with a GPU (CPU works but is slow).

## Outputs

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint by validation accuracy |
| `vocab.json` / `config.json` | Vocabulary and hyperparameters for inference |
| `training_curves.png` | Loss/accuracy curves |
| `confusion_matrix.png` | Test-set confusion matrix |

## Results

Run the notebook and record your numbers here (a model like this typically reaches roughly high-80s % test accuracy on IMDB without pretrained embeddings):

| Split | Accuracy |
|---|---|
| Validation | _TBD_ |
| Test | _TBD_ |

## Customising

- Swap the dataset: change the `kagglehub.dataset_download(...)` slug and the text/label columns.
- Go deeper or wider: edit `num_layers`, `hidden_dim` in `CFG`.
- Try LSTM: replace `nn.GRU` with `nn.LSTM` in `GatedBlock` (ignore the extra cell state on output).

## License
MIT
