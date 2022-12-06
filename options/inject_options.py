from argparse import ArgumentParser

class InjectOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--rand_select', default='Yes', type=str)
        
        self.parser.add_argument('--max_num', default=10, type=int)
        self.parser.add_argument('--batch_size', default=10, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int)

        self.parser.add_argument('--seq_weight', default=0.1, type=float)
        self.parser.add_argument('--idvec_weight', default=1.0, type=float)  
        
        self.parser.add_argument('--facenet_dir', default='./saved_models', type=str)
        self.parser.add_argument('--aadblocks_dir', default='./saved_models', type=str)
        self.parser.add_argument('--attencoder_dir', default='./saved_models', type=str)

        self.parser.add_argument('--seq_type', default='mls', type=str)
        self.parser.add_argument('--facenet_mode', default='arcface', type=str)

        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
        self.parser.add_argument('--img_dir', default='./image_datasets', type=str)

    def parse(self):
        opts = self.parser.parse_args()
        return opts