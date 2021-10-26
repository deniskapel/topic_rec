import cupy as cp
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

class TopicDataset(Dataset):
    
    def __init__(self, x, y, n_features, id2chunk, chunk2id, seq_len):
        super().__init__()
        self.data_x = x
        self.data_y = y
        self.n_features = n_features
        self.id2chunk = id2chunk
        self.seq_len = seq_len
        
    def __len__(self):
        """ returns the size of the dataset """
        return len(self.data_x)
    
    def __getitem__(self, idx):
        """ returns one sample of the dataset """
        seq_ids = self.data_x[idx]
        
        if self.seq_len == 1:
            x = self.id2chunk[seq_ids].vector
        else:
            # create an array first to avoid stacking
            x = cp.zeros((self.seq_len, self.n_features), dtype='float32')
            for i in range(self.seq_len):
                x[i] = self.id2chunk[seq_ids[i]].vector

        y = self.data_y[idx]
        return x, y, seq_ids

    def collate(self, batch):
        """
            shapes batches to pass for training
        """
        # vector 
        embeddings = cp.zeros(
             (len(batch), # batch_size
              self.seq_len, # n chunks to predict next entity
              self.n_features # vector_len
              ), dtype='float32')
        
        targets = list()
        # keep seq_ids to restore it downstream
        seq_ids = []

        for i, sample in enumerate(batch):
            embeddings[i] = sample[0]
            targets.append(sample[1])
            seq_ids.append(sample[2])

        return torch.tensor(embeddings), torch.tensor(targets), seq_ids


class GModel(nn.Module):
    def __init__(self, vocab_size, num_layers=1, seq_len=1, bidirectional=False):
        super(GModel, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = 300
        self.D = 2 if bidirectional else 1
        self.hidden_size = self.D*128
        self.num_layers=num_layers
 
        # vectors are received from Spacy noun chunks (averaged token vectors)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirectional)

        self.linear = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, prev_states: tuple=None):
        """ 
        takes tensor of prev seq and a prev state 
        returns a prediction and a new state

        x: tensor of (seq_len, batch_size, input_size)
        states: tuple of tensors (hidden_state, cell_state)
        """
        lstm_output, (state_h, state_c) = self.lstm(x)
        # pass hidden state to a fully connected layer
        logits = self.linear(lstm_output)
        return logits, (state_h, state_c)

    def init_state(self, batch_size):
        """ 
        At the beginning of each epoch hidded states has to be initialized
        both hidden state and cell state of LSTM are tensors of zeros
        """
        weight = next(self.parameters()).data

        hidden = (
            weight.new(
                self.D*self.num_layers,
                batch_size, self.hidden_size).zero_().cuda(),
            weight.new(
                self.D*self.num_layers, 
                batch_size, self.hidden_size).zero_().cuda())
        
        return hidden


def train(model, dataloader, epochs=10, lr=0.01, clip_value=1.0):

    model.train()
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # loss
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        n_batches = int(
            len(dataloader.dataset.data_x) / dataloader.batch_size)

        opt.zero_grad()
        train_losses = []
        # initialize hidden state
        state_h, state_c = model.init_state(dataloader.batch_size)
        progress_bar = tqdm(total=n_batches, desc='Epoch {}'.format(e + 1))
        
        for x, y, _ in dataloader:
            # get the output from the model
            output, (state_h, state_c) = model(x, (state_h, state_c))
            state_h = state_h.detach()
            state_c = state_c.detach()

            loss = criterion(output.transpose(1,2), y.unsqueeze(1))
            loss.backward()
            
            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            opt.step()

            train_losses.append(loss.item())
            progress_bar.update()
            progress_bar.set_postfix(train_loss = np.mean(train_losses[-100:]))

        progress_bar.close()


def predict(model, dataloader, dictionary):
    """ temp: use only the first batch """
    model.eval()
    state_h, state_c = model.init_state(dataloader.batch_size)
    softmax = nn.Softmax(dim=-1)
    
    for x, y, prev_ids in dataloader:
        preds, (state_h, state_c) = model(x, (state_h, state_c))
        state_h = state_h.detach()
        state_c = state_c.detach()
       
        preds = softmax(preds.squeeze(1)).argmax(-1)
        for y_true, y_pred in zip(y, preds):
            print('After --{}--, I suggest discussing --{}--'.format(
                    dictionary[y_true], dictionary[y_pred]))
        break