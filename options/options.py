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
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--dis_lr', default=1e-4, type=float)
        self.parser.add_argument('--gen_lr', default=1e-4, type=float)
        self.parser.add_argument('--encoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--decoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--fuser_lr', default=1e-4, type=float)
        self.parser.add_argument('--separator_lr', default=1e-4, type=float)
        self.parser.add_argument('--step_size', default=10, type=int)

        self.parser.add_argument('--conidloss_mode', default='Cos', type=str)
        self.parser.add_argument('--conrecloss_mode', default='lpips', type=str)
        self.parser.add_argument('--secfeatloss_mode', default='Cos', type=str)
        
        self.parser.add_argument('--con_att_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_id_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_rec_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_feat_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_mse_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_lpips_lambda', default=1.0, type=float)

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



class GenerateOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--secret_size', default=256, type=int)
        self.parser.add_argument('--generate_bs', default=8, type=int)
        self.parser.add_argument('--secret_bs', default=8, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        
        self.parser.add_argument('--cover_dir', default='./cover_image', type=str)
        self.parser.add_argument('--secret_dir', default='./secret_image', type=str)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)
        self.parser.add_argument('--output_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class ExtractOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--secret_size', default=256, type=int)
        self.parser.add_argument('--extract_bs', default=8, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
                        
        self.parser.add_argument('--container_dir', default='./cover_image', type=str)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)
        self.parser.add_argument('--output_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class TrainVaeOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)

        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--train_bs', default=8, type=int)
        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=10, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--encoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--decoder_lr', default=1e-4, type=float)

        self.parser.add_argument('--rec_lambda', default=1.0, type=float)
        self.parser.add_argument('--kl_lambda', default=1.0, type=float)

        self.parser.add_argument('--recloss_mode', default='l2', type=str)

        self.parser.add_argument('--board_interval', default=50, type=int)
        self.parser.add_argument('--image_interval', default=1000, type=int)
        self.parser.add_argument('--validation_interval', default=1, type=int)

        self.parser.add_argument('--train_dir', default='./train_image', type=str)
        self.parser.add_argument('--val_dir', default='./val_image', type=str)
        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class TrainPspOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)

        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--train_bs', default=8, type=int)
        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=10, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--encoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--mapper_lr', default=1e-4, type=float)
        self.parser.add_argument('--decoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--step_size', default=10, type=int)

        self.parser.add_argument('--mse_lambda', default=1.0, type=float)
        self.parser.add_argument('--lpips_lambda', default=1.0, type=float)
        self.parser.add_argument('--feat_lambda', default=1.0, type=float)

        self.parser.add_argument('--board_interval', default=50, type=int)
        self.parser.add_argument('--image_interval', default=1000, type=int)
        self.parser.add_argument('--validation_interval', default=1, type=int)

        self.parser.add_argument('--train_dir', default='./train_image', type=str)
        self.parser.add_argument('--val_dir', default='./val_image', type=str)
        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class InferencePspOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--batchsize', default=8, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)
        self.parser.add_argument('--image_dir', default='./image', type=str)
        self.parser.add_argument('--output_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class TrainDisOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)

        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--train_bs', default=8, type=int)
        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=10, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--dis_lr', default=1e-4, type=float)
        self.parser.add_argument('--gen_lr', default=1e-4, type=float)
        self.parser.add_argument('--step_size', default=10, type=int)

        self.parser.add_argument('--idloss_mode', default='Cos', type=str)
        self.parser.add_argument('--recloss_mode', default='lpips', type=str)

        self.parser.add_argument('--att_lambda', default=1.0, type=float)
        self.parser.add_argument('--id_lambda', default=1.0, type=float)
        self.parser.add_argument('--rec_lambda', default=1.0, type=float)
        self.parser.add_argument('--kl_lambda', default=1.0, type=float)

        self.parser.add_argument('--board_interval', default=50, type=int)
        self.parser.add_argument('--image_interval', default=1000, type=int)
        self.parser.add_argument('--validation_interval', default=1, type=int)

        self.parser.add_argument('--train_dir', default='./train_image', type=str)
        self.parser.add_argument('--val_dir', default='./val_image', type=str)
        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class InferenceDisOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--batchsize', default=8, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)
        self.parser.add_argument('--image_dir', default='./image', type=str)
        self.parser.add_argument('--output_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class TestDisOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--batchsize', default=1, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)
        self.parser.add_argument('--image_dir', default='./image', type=str)
        self.parser.add_argument('--output_dir', default='./experiment', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class TrainSihnOptions:
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
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--att_lr', default=1e-4, type=float)
        self.parser.add_argument('--gen_lr', default=1e-4, type=float)
        self.parser.add_argument('--fuser_lr', default=1e-4, type=float)
        self.parser.add_argument('--separator_lr', default=1e-4, type=float)
        self.parser.add_argument('--encoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--decoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--step_size', default=10, type=int)

        self.parser.add_argument('--conidloss_mode', default='Cos', type=str)
        self.parser.add_argument('--conrecloss_mode', default='lpips', type=str)
        self.parser.add_argument('--secfeatloss_mode', default='Cos', type=str)
        
        self.parser.add_argument('--con_att_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_id_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_rec_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_feat_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_mse_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_lpips_lambda', default=1.0, type=float)

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


class TrainSihnAlphaOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)

        self.parser.add_argument('--image_size', default=256, type=int)
        self.parser.add_argument('--secret_size', default=256, type=int)
        self.parser.add_argument('--train_bs', default=8, type=int)
        self.parser.add_argument('--secret_bs', default=8, type=int)
        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=10, type=int)
        self.parser.add_argument('--num_workers', default=0, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--latent_dim', default=512, type=int)
        self.parser.add_argument('--checkpoint_dir', default='./best_models', type=str)

        self.parser.add_argument('--att_lr', default=1e-4, type=float)
        self.parser.add_argument('--gen_lr', default=1e-4, type=float)
        self.parser.add_argument('--encoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--decoder_lr', default=1e-4, type=float)
        self.parser.add_argument('--fuser_lr', default=1e-4, type=float)
        self.parser.add_argument('--separator_lr', default=1e-4, type=float)

        self.parser.add_argument('--step_size', default=10, type=int)

        self.parser.add_argument('--conidloss_mode', default='Cos', type=str)
        self.parser.add_argument('--conrecloss_mode', default='lpips', type=str)
        self.parser.add_argument('--secfeatloss_mode', default='Cos', type=str)
        
        self.parser.add_argument('--con_att_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_id_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_mse_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_lpips_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_ssim_lambda', default=1.0, type=float)
        self.parser.add_argument('--con_psnr_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_feat_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_mse_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_lpips_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_ssim_lambda', default=1.0, type=float)
        self.parser.add_argument('--sec_psnr_lambda', default=1.0, type=float)

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