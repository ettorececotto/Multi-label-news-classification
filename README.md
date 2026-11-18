# Multi-label news classification

A multi-input neural network that assigns **2–3 of 18 subject categories** to short news articles, combining the article text, a Bag-of-Words vector and publication metadata.

Course project for *Artificial Neural Networks and Deep Learning*, built in Python with TensorFlow/Keras. In the written exam I had to design the solution (inputs, preprocessing, architecture, loss, evaluation); this notebook is its implementation, and the few points where it differs from my written solution are marked as `CHANGE` in the notebook. The whole pipeline (data inspection, preprocessing, model, tuning, evaluation) is in the notebook, with the reasoning for each choice written next to the code.

## Task and data

- **Dataset:** 11,000 news articles, 2012–2022, provided for the course and downloaded by the first cell of the notebook.
- **Inputs per article:** text (up to 100 words, 28 on average), a 10,000-dimensional Bag-of-Words vector, year and month.
- **Target:** 18 non-exclusive categories (each article has 2–3 labels; classes range from about 1,000 to 4,000 articles).

## Approach

**Split.** Random 80/10/10 train/validation/test split (8,800 / 1,100 / 1,100), fixed seed (213). The split is done *before* preprocessing, so the vocabulary and the scalers are fit on the training set only, with no leakage. A stratified split was not possible with plain scikit-learn because the target is multi-label.

**Preprocessing** (one pipeline per input):
- Text: lowercasing, punctuation removal, vocabulary pruning (rare words and the 15 most frequent words mapped to `<UNK>`), integer encoding, padding/truncation to 28 tokens.
- Bag-of-Words and year: Min-Max scaling.
- Month: one-hot encoding (the month has no meaningful order).

**Model.** Four branches concatenated into one classifier (about 1.8M parameters):
- Text → Embedding → Bidirectional LSTM
- Bag-of-Words → MLP (3 Dense layers with Dropout)
- Year and month → concatenated directly
- Concatenation → Dense(64) → Dropout → Dense(18, sigmoid)

Sigmoid outputs and binary cross-entropy are used because the labels are not mutually exclusive. Optimizer: Adam (lr = 0.001), dropout 0.3, batch size 64.

**Tuning.** Grid search over embedding size {50, 100}, LSTM units {64, 128} and BoW layer size {128, 256} (8 configurations), selected on validation macro-F1. The search was kept small to fit the exam's time limit. Best configuration: embedding 50, LSTM 128, BoW 128.

**Training.** Early Stopping on validation loss (patience 5) with the best weights restored (epoch 3).

## Results (held-out test set, threshold 0.5)

| Metric (macro-average) | Value |
| --- | --- |
| F1-score | 0.522 |
| Precision | 0.797 |
| Recall | 0.440 |
| Accuracy | 0.917 |

Accuracy is misleading here, since each category is absent from most articles, so macro-F1 is the main metric. The notebook also shows per-class metrics and one confusion matrix per label.

## Limitations and next steps

- Precision is much higher than recall: with a fixed 0.5 threshold the model is cautious and misses many true labels. Tuning the decision threshold per class would be a natural next step.
- Early Stopping on validation loss restored an early epoch (3). Validation macro-F1 kept improving for a few more epochs, so monitoring macro-F1 instead could give better results.
- A multi-label stratified split (for example with `iterative-stratification`) would make the class proportions more consistent across the three sets.
- Only a small grid was explored; a wider search was out of scope for the time limit.

## How to run

The notebook is written for Google Colab: open it there and run all cells. To run it locally:

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

`plot_model` also needs `pydot` and Graphviz; skip that cell if you don't need the architecture diagram. The first cell downloads the dataset with `wget`, so on Windows replace it with a manual download.
