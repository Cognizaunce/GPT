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
        

DATA_PATH = "./data/input.txt"
BLOCK_SIZE = 128 # spatial context of the transformer

file = open(DATA_PATH, "r")
data = file.read(-1) # -1 means read the whole file. If file size is large, you may want to consider replacing it with the number of characters to be read.
dataset = CharacterDataset(data = data, block_size = BLOCK_SIZE)

batch = dataset[1] # returns a tuple of tensors at idx = 0. Feel free to chnage the idx
x, y = batch
x = x.tolist()
y = y.tolist()

for i in range(len(x)):
    x[i] = dataset.itos[x[i]]
    y[i] = dataset.itos[y[i]]

print(f"\033[1mTraining Data : \033[0m\n\n{''.join(x)}")
print(f"\n\033[1mTargets : \033[0m\n\n{''.join(y)}")

gpt_config = GPTconfig(num_layers = 2, 
                       n_heads = 12, 
                       embd_size = 768, 
                       vocab_size = dataset.vocab_size, 
                       block_size = dataset.block_size
                      )

model = GPT(gpt_config)

train_config = TrainingConfig(max_epochs = 2, 
                              batch_size = 256, 
                              lr_decay = True, 
                              lr = 6e-4,
                              warmup_tokens = 512*20,
                              final_tokens = 2 * len(dataset) * dataset.block_size,
                              ckpt_path = "./checkpoints/transformers.pt"
                             )

trainer = Trainer(model = model, train_set = dataset, test_set = None, configs = train_config)
trainer.train()

seed_context = "Help me!"
x = torch.tensor([dataset.stoi[s] for s in seed_context], dtype=torch.long, device=trainer.device)[None,...]
y = sample_context(model=model, x=x, steps=10000, temperature=1.0, sample=True, top_k=10)[0]

y = y.tolist()
y = [dataset.itos[i] for i in y]
y = "".join(y)

print(f"\033[1mGenerated Data : \033[0m\n\n{y}")