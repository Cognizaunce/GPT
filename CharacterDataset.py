import torch
import numpy as np

from gpt import GPT, GPTconfig
from gpt import TrainingConfig, Trainer
from gpt import sample_context
# Could have imported all of them at once. Doesn't matter :)

from gpt.utils.utils import seed_all
# Set all random seeds to 0 for reproducibility
seed_all(0)

from torch.utils.data import Dataset as Dataset

class CharacterDataset(Dataset):
        def __init__(self, data, block_size):
            characters = sorted(list(set(data)))
            data_size, vocab_size = len(data), len(characters)
            
            print(f"Dataset has {data_size} characters. {vocab_size} of characters are unique.")
            
            # char to idx mapping and vice-versa
            self.stoi = {ch:i for i,ch in enumerate(characters)}
            self.itos = {i:ch for i,ch in enumerate(characters)}
            
            self.block_size = block_size
            self.vocab_size = vocab_size
            self.data_size = data_size
            self.data = data
            
        def __len__(self):
            return self.data_size - self.block_size
        
        def __getitem__(self, idx):
            # take a chunk of data from the given index from the dataset
            chunk = self.data[idx : idx + self.block_size + 1]
            
            #convert the chunk to integers
            data = [self.stoi[ch] for ch in chunk]
            
            # create x and y. 
            # x will contain every but the last character in the chunk.
            # y will contain every but the first character in the chunk.
            # Hence this will create an offset in targets by 1.
            # Thus helps in language modelling. Given a character, the goal of the transformer would be to predict the next character in sequence.
            
            x = torch.tensor(data[:-1], dtype=torch.long) # nn.Embedding requires input data to be in torch.long
            y = torch.tensor(data[1:], dtype=torch.long)
            
            return x,y