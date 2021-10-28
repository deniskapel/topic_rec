from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

class Chunker():
    # Class to extract noun chunks from dialogues

    def __init__(self, spacy_model, stop_words):
        self.counter = Counter()
        self.stops = stop_words
        self.nlp = spacy_model

    def normalize(self, dialogues: list):
        """ iterates over utterances of a dialogue and shapes the output """
        dials = [self.parse_by_utterance(d) for d in tqdm(dialogues)]
        return [d for d in dials if d]

    def __extract_chunks(self, chunked_utterance: list) -> list:
        """
            extract topic and necessary chunk information from
            every meaningfull noun chunk in the utternace
        """
        output = []
        for chunk in chunked_utterance:
            topic = chunk.lower_
            if topic in self.stops:
                continue
            self.counter.update([topic])

            output.append(chunk)
        return output

    def parse_by_utterance(self, dialogue:str, delim:str = '__eou__') -> list:
        """
            use spacy parser to extract noun_chunks
            parse a dialogue by utterance to improve extraction

            returns list of Spacy Span instance
        """
        # split for better parsing
        parsed = [self.nlp(ut.strip()) for ut in dialogue.split(delim)]
        # merge noun_chunks of each utterance into one dialogue
        output = chain(*[self.__extract_chunks(ut.noun_chunks) for ut in parsed])
        # return only filled utterances, as some of them have stop words only
        return [ut for ut in output if ut]

    def filter_chunks(self, dialogue: list, to_remove: list):
        """
            filter out noun chunks
            to_remove s is a list of chunks to remove, e.g. top most common
        """
        return [chunk for chunk in dialogue if chunk.lower_ not in to_remove]


class Vectorizer():
    """
    Class to vectorize dialogues best on the vector of the chunks last word
    dialogue is an averaged vector of all noun chunks [their last word]
    """

    def __init__(self, num_docs: int):
        self.num_docs = num_docs

    def vectorize(self, docs: list):
        """  indexes docs and return vector for each doc """
        vec = np.zeros((self.num_docs, 300))
        doc2id = pd.DataFrame({'doc': docs})

        for i, doc in enumerate(docs):
            vec[i] = self.__data_generation(doc)

        return vec, doc2id

    def __data_generation(self, doc) -> tuple:
        """
            parse doc to return a tuple topics, vector
            returns list of topics
        """
        vec = np.zeros((len(doc), 300), dtype='float32')
        topics = []
        for i, chunk in enumerate(doc):
            topics.append(chunk.lower_)
            vec[i] = chunk.vector.get()

        # return an averaged vector for all noun chunks for clusterisation
        return np.mean(vec, 0)


class SequenceGenerator():
    """
    Class to generate prev2next sequences of chunks
    if some seq of different sized, leave only the ones of the same size
    or pad
    seq_len: int indicating the length of the sequence to use for prediction
    chunk2id and id2chunk: mappings of spaCy tokens and their index.

    Use the same chunk2id and id2chunk that you will pass to TokenDataset.
    OOV is handled by spacy and this dictionary is used only to extract vectors
    """
    def __init__(
        self, seq_len: int, chunk2id: dict, id2chunk: list):
        # lengths of the sequence to predict the next
        self.seq_len = seq_len
        self.chunk2id = chunk2id
        self.id2chunk = id2chunk

    def get_sequences(self, docs: pd.DataFrame) -> pd.DataFrame:
        """
        function generating sequences from rows of pandas dataset

        docs: each or is a list of tokens (str, Spacy noun chunks, etc.)
        update_vocab: boolean controlling whether to add new vocabulary or not
        """
        seqs = []
        seqs.extend(docs['doc'].apply(self.generate_seq))
        seqs = [seq for seq in chain(*seqs)]

        # replace chunks with their ids
        sequences = []
        for seq in seqs:
            # handle oov
            ids = [self.chunk2id.get(chunk_id, 0) for chunk_id in seq]
            sequences.append([ids[:-1], ids[1:]])

        return pd.DataFrame(sequences, columns=['seq', 'target'])


    def generate_seq(self, row) -> tuple:
        """
        from a given doc generates a sequence of prev tokens and the next one
        """
        sequences = []

        # if the number of tokens > seq_len
        if len(row) > self.seq_len:
            for i in range(self.seq_len, len(row)):
                # select sequence of tokens
                seq = row[i-self.seq_len:i+1]
                # add to the list
                sequences.append(seq)

            return sequences
        # return unprocessed if n_tokens == seq_len
        return row
