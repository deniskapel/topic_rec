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
    """ 
    def __init__(self, seq_len: int):
        # lengths of the sequence to predict the next
        self.seq_len = seq_len
        self.chunk2id = {}
        self.id2chunk = []

    def get_sequences(self, docs: pd.DataFrame, seq_len = 3) -> pd.DataFrame:
        seqs = []
        seqs.extend(docs['doc'].apply(self.generate_seq))
        seqs = [seq for seq in chain(*seqs)]
        self.id2chunk = list(set([chunk for chunk in chain(*seqs)]))
        self.chunk2id = {chunk: i for i, chunk in enumerate(self.id2chunk)}
        # replace chunks with their ids
        sequences = []
        for seq in seqs:
            # to be modified for pre-sequences larger than 1
            pre_ids = [self.chunk2id[chunk_id] for chunk_id in seq[:-1]]
            post_ids = [self.chunk2id[chunk_id] for chunk_id in seq[1:]]
            
            sequences.append([pre_ids, post_ids])
        
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
