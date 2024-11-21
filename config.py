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
        self.model_path = ""  # empty for training from begining
        self.benchmark_list = [
                'adaptec1', 
                # 'adaptec2', 'adaptec3', 'adaptec4',
                'bigblue1', 
                # 'bigblue2', 'bigblue3', 'bigblue4', 
                'ibm01', 
                # 'ibm02', 'ibm03', 'ibm04'
                ]
config = Config()