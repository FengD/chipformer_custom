# config.py
# description: define all the parameters


class Config:
    def __init__(self):
        self.grid = 84
        # gpt-2 param
        self.num_layers = 6
        self.num_heads = 8
        self.embedding_dim = 128
        self.seq_len = 256
        self.seed = 42

config = Config()