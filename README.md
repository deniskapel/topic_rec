# Next Entity Recommendation using Spacy and LSTM

Next entity is a noun chunk extracted by Spacy. Using a sequence of previous noun chunks, predict the next one.

Model contain LSTM layer and a fully connected one. Vectors are extracted from Spacy instances. If a chunk consists from several tokens, its vector is the average of them.

The pipeline can be reporoduced in Colab, see [topic_rec notebook](topic_rec.ipynb) for details. In the notebook, it is described in mode details.

TODO: allow switching between bidirectional and forwad-only LSTM
