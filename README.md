# Next Entity Recommendation using Spacy and LSTM

Next entity is a noun chunk extracted by Spacy. Using a sequence of previous noun chunks, predict the next one.

Model contain LSTM layer and a fully connected one. Vectors are extracted from Spacy instances. If a chunk consists from several tokens, its vector is the average of them.

The pipeline can be reporoduced in Colab, see [topic_rec notebook](topic_rec.ipynb) for details.

Pipeline:

1. Tokenizing with Spacy. DailyDialogue Dataset has a delimiter indicating the end of utterance. Each dialogue is split into a list of utterence for better tokenisation. Each token is a Noun Chunk. When tokenized, utterances are merged into a dialogue. Top N most frequent chunks are removed from each dialogue as they do not bring any additional meaning to the dialogues.

    - To limit a range of recommendations in the future, dialogues are clustered using spaCy chunk vectors. Chunk vector is an average vector of each token in this chunk. Each dialogue is an average vector of its chunks.
    - Clusering with KMeans. The largest cluster is then taken for training.
    - CLustering smaller amounts of dialogues produces fine results, dialogues seem to be grouped by more specific topics

2. Generating sequences from dialogues. Each dialogue is parsed to create a sequance of chunks to use for prediction. SequenceGenerator creates a vocabulary list. Each dialogue set can have its own vocab list. OOV are handled by spaCy and this chunk2id is used only to extract vectors by the TopicDataset. Make sure TopicDataset and SequenceGenerator use one chunk2id dictionary.

3. Transforming sequences to TorchDataset. Embeddings are already extracted by spacy so the Dataset will transform indexes into vectors and then pass it to the model. OOV are once again handled by spaCy.

4. Training the model on spaCy embeddings. Forward LSTM and a fully connected layer are used.

5. For the demosntration purposes, predict() is using both original dict and the one generated for the new dataset. It will allow to compare the original tokens, predictions and targets.

TODO: refactor to allow switching between bidirectional and forwad-only LSTM
