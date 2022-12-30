import os
import numpy as np
from PIL import Image
from scipy import spatial 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

#from thop import profile
#from thop import clever_format


def alignment(images):
  return F.interpolate(images[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True)


def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def tensor2img(var):
    # var: 3 x 256 x 256 --> 256 x 256 x 3
    var = var.cpu().detach().numpy().transpose([1,2,0])
    # de-normalize
    #var = ((var+1) / 2) 
    mean, std = 0.5, 0.5
    var = std * var + mean
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))



def visualize_results(vis_dict, dis_num, epoch, prefix, save_dir, iter=None, step=None):
    if prefix == 'train':
        ResultImgName = os.path.join(save_dir, 'ResultPics_epoch{:05d}_iter{:05d}_step{:05d}.png'.format(epoch, iter, step))
    elif prefix == 'validation':
        ResultImgName = os.path.join(save_dir, 'ResultPics_epoch{:05d}.png'.format(epoch))
    elif prefix == 'best':
        ResultImgName = os.path.join(save_dir, 'BestResultPics.png')
    else:
        raise ValueError('[*]Invalid result picture save prefix. Must be one of train, validation or best')

    cover_gap = vis_dict['container'] - vis_dict['cover']
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)

    secret_gap = vis_dict['secret_rec'] - vis_dict['secret_input']
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(44, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=11)
    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        cover = tensor2img(vis_dict['cover'][img_idx])
        plt.imshow(cover)
        plt.title('Cover')

        fig.add_subplot(gs[img_idx, 1])
        container = tensor2img(vis_dict['container'][img_idx]) 
        plt.imshow(container)
        plt.title('Container')

        fig.add_subplot(gs[img_idx, 2])
        covgap_img = tensor2img(cover_gap[img_idx])
        plt.imshow(covgap_img)
        plt.title('Cover Gap')

        fig.add_subplot(gs[img_idx, 3])
        secret = tensor2img(vis_dict['secret_input'][img_idx])
        plt.imshow(secret)
        plt.title('Secret')

        fig.add_subplot(gs[img_idx, 4])
        secret_rec = tensor2img(vis_dict['secret_rec'][img_idx])
        plt.imshow(secret_rec)
        plt.title('Secret_rec')

        fig.add_subplot(gs[img_idx, 5])
        secgap_img = tensor2img(secret_gap[img_idx])
        plt.imshow(secgap_img)
        plt.title('Secret Gap')

        fig.add_subplot(gs[img_idx, 6])
        secret_feature = vis_dict['secret_feature_input'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature)
        plt.grid()
        plt.title('Secret Feature Ori')

        fig.add_subplot(gs[img_idx, 7])
        cover_id = vis_dict['cover_id'][img_idx].cpu().detach().numpy()
        plt.plot(cover_id)
        plt.grid()
        plt.title('Cover Id')

        fig.add_subplot(gs[img_idx, 8])
        input_feature = vis_dict['input_feature'][img_idx].cpu().detach().numpy()
        plt.plot(input_feature)
        plt.grid()
        plt.title('Input feature')

        fig.add_subplot(gs[img_idx, 9])
        container_id = vis_dict['container_id'][img_idx].cpu().detach().numpy()
        plt.plot(container_id)
        plt.grid()
        cover_similarity = 1 - spatial.distance.cosine(cover_id, container_id)
        fused_similarity = 1 - spatial.distance.cosine(input_feature, container_id)
        plt.title('Container_id CovSim:{:.2f} FusSim:{:.2f}'.format(cover_similarity, fused_similarity))

        fig.add_subplot(gs[img_idx, 10])
        secret_feature_rec = vis_dict['secret_feature_rec'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_rec)
        plt.grid()
        feat_similarity = 1 - spatial.distance.cosine(secret_feature + 1e-5, secret_feature_rec)
        plt.title('Secret Feat Rec FeatSim:{:.2f}'.format(feat_similarity))

    plt.tight_layout()
    fig.savefig(ResultImgName)
    plt.close(fig)



def visualize_vae_results(vis_dict, dis_num, epoch, prefix, save_dir, iter=None, step=None):
    if prefix == 'train':
        ResultImgName = os.path.join(save_dir, 'ResultPics_epoch{:05d}_iter{:05d}_step{:05d}.png'.format(epoch, iter, step))
    elif prefix == 'validation':
        ResultImgName = os.path.join(save_dir, 'ResultPics_epoch{:05d}.png'.format(epoch))
    elif prefix == 'best':
        ResultImgName = os.path.join(save_dir, 'BestResultPics.png')
    else:
        raise ValueError('[*]Invalid result picture save prefix. Must be one of train, validation or best')

    rec_gap = vis_dict['image_ori'] - vis_dict['image_rec']
    rec_gap = (rec_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(12, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=3)
    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        cover = tensor2img(vis_dict['image_ori'][img_idx])
        plt.imshow(cover)
        plt.title('Image original')

        fig.add_subplot(gs[img_idx, 1])
        container = tensor2img(vis_dict['image_rec'][img_idx]) 
        plt.imshow(container)
        plt.title('Image reconstructed')

        fig.add_subplot(gs[img_idx, 2])
        covgap_img = tensor2img(rec_gap[img_idx])
        plt.imshow(covgap_img)
        plt.title('Image Rec Gap')

    plt.tight_layout()
    fig.savefig(ResultImgName)
    plt.close(fig)



def visualize_correlation(seq, id, img, save_path):
    seq_len = len(seq)
    
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(nrows=1, ncols=2)

    fig.add_subplot(gs[0, 0])
    img = tensor2img(img)
    plt.imshow(img)
    plt.title('Input Image')
    
    fig.add_subplot(gs[0, 1])
    id = id.to(torch.device('cpu')).detach().numpy()
    correlation = np.correlate(id, seq, 'full')
    plt.plot(np.arange(-seq_len+1, seq_len), correlation, '.-')
    plt.grid()
    plt.title('Correlation')

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig



def statistic_correlation(id_vector, seq):
    seq_len = len(seq)

    peak_list = []
    avg_list = []

    for idx in range(len(id_vector)):
        id_vec = id_vector[idx].to(torch.device('cpu')).detach().numpy()
        
        corr_abs = np.abs(np.correlate(id_vec, seq, 'full'))

        peak = corr_abs[seq_len-1]

        corr_delpeak = np.delete(corr_abs, obj=seq_len-1, axis=None)

        avg = np.mean(corr_delpeak)

        peak_list.append(peak)
        avg_list.append(avg)
        
    return peak_list, avg_list



def calculatie_correlation(id_vector, seq, threshold):
    seq_len = len(seq)

    verify_list = []

    for idx in range(len(id_vector)):
        id_vec = id_vector[idx].to(torch.device('cpu')).detach().numpy()

        corr_abs = np.abs(np.correlate(id_vec, seq, 'full'))

        peak = corr_abs[seq_len-1]

        corr_delpeak = np.delete(corr_abs, obj=seq_len-1, axis=None)

        avg = np.mean(corr_delpeak)

        if peak/avg >= threshold:
            verify_list.append(1)
        else:
            verify_list.append(0)

    return verify_list



def evaluation(label, pred):
    
    accuracy = accuracy_score(y_true=label, y_pred=pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true=label, y_pred=pred, average='binary')

    tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=pred).ravel()

    return accuracy, precision, recall, f1_score, tn, fp, fn, tp



"""
def count_models(test_model, dummy_input):
    flops, params = profile(test_model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], '%.4f')

    return flops, params
"""



def print_log(info, log_path, console=True):
    ##### Print the information into the console #####
    if console:
        print(info)
    ##### Write the information into the log file #####
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_path + "\n")
    else:
        with open(log_path, "a+") as f:
            f.writelines(info + "\n")



def log_metrics(writer, data_dict, step, prefix):
    for key, value in data_dict.items():
        writer.add_scalar('{}/{}'.format(prefix, key), value, (step))



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count