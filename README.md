# Next Entity Recommendation using spaCy and LSTM

Next entity is a noun chunk extracted by spaCy. Using a sequence of previous noun chunks, predict the next one.

The model contains an LSTM layer and a fully connected one. Vectors are extracted from spaCy instances.

The pipeline can be reproduced using Colab, see [topic_rec notebook](topic_rec.ipynb) for details.

Pipeline:

1. **Tokenizing with spaCy**. [The DailyDialogue Dataset](http://yanran.li/dailydialog) uses a delimiter separating utterances. Each dialogue is split into a list of utterences to improve tokenization. Each token is a Noun Chunk. When tokenized, utterances are grouped by dialogue. Top N most frequent chunks can be removed from each dialogue as they do not bring any additional meaning to the dialogues. Potentially, very rare tokens (freq=1) can be removed as well but the results' comparison is yet to be investigated.

    - To limit a range of recommendations, dialogues are clustered using spaCy chunk vectors. A chunk vector is an average vector of each token in this chunk. Each dialogue is an average vector of its chunks.
    - Clusering with KMeans. The largest cluster is then taken for training.
    - CLustering smaller amounts of dialogues shows fine results, dialogues seem to be grouped by more specific topics

2. **Generating sequences from dialogues**. 

Each dialogue is parsed to create a sequance of chunks to use for prediction using the vocab2id and id2vocab mappings. OOV are handled by spaCy and a chunk id is used only to extract vectors by the TopicDataset downstream. Make sure instances of TopicDataset and SequenceGenerator classes use the same chunk2id dictionary.

3. **Transforming sequences to a Torch Dataset**. 

Embeddings are already created by spaCy, and the TopicDataset will transform chunk ids into vectors and then pass it to the model.

4. Training the model on spaCy embeddings. Forward LSTM and a fully connected layer are used.

5. For the demosntration purposes, predict() is using both original dict and the one generated for the new dataset. It enables the comparison of the original sequence, predictions and targets.


To explore in the future:

* Cluster themes and save centroids
* Train smaller models for each theme
* When a new sample comes, based on its vector and the centroids, find the most relevant theme and use its model
* Compare the results of out-of-domain cases with outputs given by a single model only

TODO: refactor to allow switching between bidirectional and forwad-only LSTM or use a more complex model
