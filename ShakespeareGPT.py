# filename: ShakespeareGPT.py

import torch
import numpy as np

from gpt import GPT, GPTconfig, TrainingConfig, Trainer, sample_context
from CharacterDataset import CharacterDataset

import argparse

def main(name="Python"):
      
    data = open("./data/input.txt", "r").read(-1)
    dataset = CharacterDataset(data = data, block_size = 128)

    gpt_config = GPTconfig(num_layers = 2, n_heads = 12, embd_size = 768, vocab_size = dataset.vocab_size, block_size = dataset.block_size)

    model = GPT(gpt_config)
    model.load_state_dict(torch.load("./checkpoints/transformers.pt"))

    train_config = TrainingConfig(max_epochs = 2, batch_size = 256, lr_decay = True, lr = 6e-4, warmup_tokens = 512*20, final_tokens = 2 * len(dataset) * dataset.block_size, ckpt_path = "./checkpoints/transformers.pt")

    trainer = Trainer(model = model, train_set = dataset, test_set = None, configs = train_config)

    seed_context = name
    x = torch.tensor([dataset.stoi[s] for s in seed_context], dtype=torch.long, device=trainer.device)[None,...]
    y = sample_context(model=model, x=x, steps=3000, temperature=1.0, sample=True, top_k=10)[0]

    y = y.tolist()
    y = [dataset.itos[i] for i in y]
    y = "".join(y)

    print(f"\033[1mGenerated Data : \033[0m\n\n{y}")


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="My Script with Command-Line Arguments")

# Add a command-line argument for the name parameter
parser.add_argument('--name', default="Python", help="Specify the name for the greeting")

# Parse the command-line arguments
args = parser.parse_args()

# Call the main function with the specified name
if __name__ == "__main__":
    main(args.name)
