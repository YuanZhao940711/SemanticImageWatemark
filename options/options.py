from argparse import ArgumentParser



class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--train_bs', default=8, type=int)
        self.parser.add_argument('--secret_bs', default=8, type=int)
        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=10, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        
        self.parser.add_argument('--facenet_dir', default='./saved_models', type=str)
        self.parser.add_argument('--aadblocks_dir', default='./saved_models', type=str)
        self.parser.add_argument('--attencoder_dir', default='./saved_models', type=str)
        self.parser.add_argument('--discriminator_dir', default='./saved_models', type=str)
        self.parser.add_argument('--fuser_dir', default='./saved_models', type=str)
        self.parser.add_argument('--separator_dir', default='./saved_models', type=str)
        self.parser.add_argument('--encoder_dir', default='./saved_models', type=str)
        self.parser.add_argument('--decoder_dir', default='./saved_models', type=str)

        self.parser.add_argument('--lr_aad', default=1e-4, type=float)
        self.parser.add_argument('--lr_att', default=1e-4, type=float)
        self.parser.add_argument('--lr_dis', default=1e-4, type=float)
        self.parser.add_argument('--lr_fuser', default=1e-4, type=float)
        self.parser.add_argument('--lr_separator', default=1e-4, type=float)
        self.parser.add_argument('--lr_encoder', default=1e-4, type=float)
        self.parser.add_argument('--lr_decoder', default=1e-4, type=float)

        self.parser.add_argument('--adv_lambda', default=1.0, type=float)
        self.parser.add_argument('--att_lambda', default=1.0, type=float)
        self.parser.add_argument('--id_lambda', default=1.0, type=float)
        self.parser.add_argument('--rec_con_lambda', default=1.0, type=float)
        self.parser.add_argument('--rec_sec_lambda', default=1.0, type=float)
        self.parser.add_argument('--feat_lambda', default=1.0, type=float)

        self.parser.add_argument('--idloss_mode', default='Cos', type=str)
        self.parser.add_argument('--recloss_mode', default='lpips', type=str)
        self.parser.add_argument('--featloss_mode', default='MAE', type=str)
        self.parser.add_argument('--facenet_mode', default='arcface', type=str)

        self.parser.add_argument('--board_interval', default=50, type=int)
        self.parser.add_argument('--image_interval', default=1000, type=int)
        self.parser.add_argument('--validation_interval', default=1, type=int)

        self.parser.add_argument('--train_cover_dir', default='./train_cover_image', type=str)
        self.parser.add_argument('--val_cover_dir', default='./val_cover_image', type=str)
        self.parser.add_argument('--secret_dir', default='./secret_image', type=str)
        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts