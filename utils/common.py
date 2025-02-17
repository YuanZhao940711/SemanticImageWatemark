import os
import numpy as np
from PIL import Image
from scipy import spatial 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0) 
        m.bias.data.fill_(0)



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

    secret_gap = vis_dict['secret_output'] - vis_dict['secret_ori']
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(55, 5*dis_num))
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
        secret = tensor2img(vis_dict['secret_ori'][img_idx])
        plt.imshow(secret)
        plt.title('Secret')

        fig.add_subplot(gs[img_idx, 4])
        secret_rec = tensor2img(vis_dict['secret_output'][img_idx])
        plt.imshow(secret_rec)
        plt.title('Secret_rec')

        fig.add_subplot(gs[img_idx, 5])
        secgap_img = tensor2img(secret_gap[img_idx])
        plt.imshow(secgap_img)
        plt.title('Secret Gap')

        fig.add_subplot(gs[img_idx, 6])
        secret_feature_input = vis_dict['secret_feature_input'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_input)
        plt.grid()
        plt.title('Secret_Feature_Input')

        fig.add_subplot(gs[img_idx, 7])
        cover_id = vis_dict['cover_id'][img_idx].cpu().detach().numpy()
        plt.plot(cover_id)
        plt.grid()
        plt.title('Cover_Id')

        fig.add_subplot(gs[img_idx, 8])
        input_feature = vis_dict['input_feature'][img_idx].cpu().detach().numpy()
        plt.plot(input_feature)
        plt.grid()
        features_similarity = 1 - spatial.distance.cosine(secret_feature_input + 1e-5, input_feature)
        coverfeat_similarity = 1 - spatial.distance.cosine(cover_id, input_feature)
        plt.title('Input_feature FeatsSim:{:.2f} CovFeatSim:{:.2f}'.format(features_similarity, coverfeat_similarity))

        fig.add_subplot(gs[img_idx, 9])
        container_id = vis_dict['container_id'][img_idx].cpu().detach().numpy()
        plt.plot(container_id)
        plt.grid()
        cover_similarity = 1 - spatial.distance.cosine(cover_id, container_id)
        fused_similarity = 1 - spatial.distance.cosine(input_feature, container_id)
        plt.title('Container_id CovSim:{:.2f} FusSim:{:.2f}'.format(cover_similarity, fused_similarity))

        fig.add_subplot(gs[img_idx, 10])
        secret_feature_output = vis_dict['secret_feature_output'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_output)
        plt.grid()
        feat_similarity = 1 - spatial.distance.cosine(secret_feature_input + 1e-5, secret_feature_output + 1e-5)
        plt.title('Secret_Feature_Output FeatSim:{:.2f}'.format(feat_similarity))

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

    fig = plt.figure(figsize=(16, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=4)
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

        fig.add_subplot(gs[img_idx, 3])
        input_feature = vis_dict['image_feature'][img_idx].cpu().detach().numpy()
        plt.plot(input_feature)
        plt.grid()
        plt.title('Image feature')

    plt.tight_layout()
    fig.savefig(ResultImgName)
    plt.close(fig)



def visualize_psp_results(vis_dict, dis_num, epoch, prefix, save_dir, iter=None, step=None):
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

    fig = plt.figure(figsize=(28, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=7)
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

        fig.add_subplot(gs[img_idx, 3])
        image_feature_ori = vis_dict['image_feature_ori'][img_idx].cpu().detach().numpy()
        plt.plot(image_feature_ori)
        plt.grid()
        plt.title('Image original feature')

        fig.add_subplot(gs[img_idx, 4])
        image_feature_ori_norm = vis_dict['image_feature_ori_norm'][img_idx].cpu().detach().numpy()
        plt.plot(image_feature_ori_norm)
        plt.grid()
        plt.title('Image original feature norm')

        fig.add_subplot(gs[img_idx, 5])
        image_feature_rec = vis_dict['image_feature_rec'][img_idx].cpu().detach().numpy()
        plt.plot(image_feature_rec)
        plt.grid()
        plt.title('Image reconstructed feature')

        fig.add_subplot(gs[img_idx, 6])
        image_feature_rec_norm = vis_dict['image_feature_rec_norm'][img_idx].cpu().detach().numpy()
        plt.plot(image_feature_rec_norm)
        plt.grid()
        plt.title('Image reconstructed feature norm')

    plt.tight_layout()
    fig.savefig(ResultImgName)
    plt.close(fig)



def visualize_dis_results(vis_dict, dis_num, epoch, prefix, save_dir, iter=None, step=None):
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

    fig = plt.figure(figsize=(35, 5*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=7)
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

        fig.add_subplot(gs[img_idx, 3])
        image_ori_id = vis_dict['image_ori_id'][img_idx].cpu().detach().numpy()
        plt.plot(image_ori_id)
        plt.grid()
        plt.title('Image original id')

        fig.add_subplot(gs[img_idx, 4])
        image_ori_id_norm = vis_dict['image_ori_id_norm'][img_idx].cpu().detach().numpy()
        plt.plot(image_ori_id_norm)
        plt.grid()
        plt.title('Image original id norm')

        fig.add_subplot(gs[img_idx, 5])
        image_rec_id = vis_dict['image_rec_id'][img_idx].cpu().detach().numpy()
        plt.plot(image_rec_id)
        plt.grid()
        plt.title('Image reconstructed id')

        fig.add_subplot(gs[img_idx, 6])
        image_rec_id_norm = vis_dict['image_rec_id_norm'][img_idx].cpu().detach().numpy()
        plt.plot(image_rec_id_norm)
        plt.grid()
        id_similarity = 1 - spatial.distance.cosine(image_rec_id_norm, image_ori_id_norm)
        plt.title('Image reconstructed id norm IdSim:{:.2f}'.format(id_similarity))

    plt.tight_layout()
    fig.savefig(ResultImgName)
    plt.close(fig)



def visualize_distest_results(vis_dict, save_dir):
    rec_gap = vis_dict['image_ori'] - vis_dict['image_rec']
    rec_gap = (rec_gap*10 + 0.5).clamp_(0.0, 1.0)

    dis_num = vis_dict['image_ori'].shape[0]

    fig = plt.figure(figsize=(25, 5*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=5)

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

        fig.add_subplot(gs[img_idx, 3])
        image_ori_id = vis_dict['image_ori_id'][img_idx].cpu().detach().numpy()
        plt.plot(image_ori_id)
        plt.grid()
        plt.title('Image original id')

        fig.add_subplot(gs[img_idx, 4])
        image_rec_id = vis_dict['image_rec_id'][img_idx].cpu().detach().numpy()
        plt.plot(image_rec_id)
        plt.grid()
        plt.title('Image reconstructed id')

    plt.tight_layout()
    fig.savefig(save_dir)
    plt.close(fig)



def visualize_sihn_results(vis_dict, dis_num, epoch, prefix, save_dir, iter=None, step=None):
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

    secret_rec_gap = vis_dict['secret_rec'] - vis_dict['secret_ori']
    secret_rec_gap = (secret_rec_gap*10 + 0.5).clamp_(0.0, 1.0)

    secret_output_gap = vis_dict['secret_output'] - vis_dict['secret_ori']
    secret_output_gap = (secret_output_gap*10 + 0.5).clamp_(0.0, 1.0)

    fig = plt.figure(figsize=(70, 5*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=14)
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
        secret = tensor2img(vis_dict['secret_ori'][img_idx])
        plt.imshow(secret)
        plt.title('Secret')

        fig.add_subplot(gs[img_idx, 4])
        secret_rec = tensor2img(vis_dict['secret_rec'][img_idx])
        plt.imshow(secret_rec)
        plt.title('Secret_rec')

        fig.add_subplot(gs[img_idx, 5])
        secrecgap_img = tensor2img(secret_rec_gap[img_idx])
        plt.imshow(secrecgap_img)
        plt.title('Secret_rec Gap')

        fig.add_subplot(gs[img_idx, 6])
        secret_output = tensor2img(vis_dict['secret_output'][img_idx])
        plt.imshow(secret_output)
        plt.title('Secret_output')

        fig.add_subplot(gs[img_idx, 7])
        secoutputgap_img = tensor2img(secret_output_gap[img_idx])
        plt.imshow(secoutputgap_img)
        plt.title('Secret_output Gap')

        fig.add_subplot(gs[img_idx, 8])
        cover_id = vis_dict['cover_id'][img_idx].cpu().detach().numpy()
        plt.plot(cover_id)
        plt.grid()
        plt.title('Cover_Id')
        
        fig.add_subplot(gs[img_idx, 9])
        secret_feature_input = vis_dict['secret_feature_input'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_input)
        plt.grid()
        plt.title('Secret_Feature_Input')

        fig.add_subplot(gs[img_idx, 10])
        input_feature = vis_dict['input_feature'][img_idx].cpu().detach().numpy()
        plt.plot(input_feature)
        plt.grid()
        features_similarity = 1 - spatial.distance.cosine(secret_feature_input + 1e-5, input_feature)
        coverfeat_similarity = 1 - spatial.distance.cosine(cover_id, input_feature)
        plt.title('Input_feature FeatsSim:{:.2f} CovFeatSim:{:.2f}'.format(features_similarity, coverfeat_similarity))

        fig.add_subplot(gs[img_idx, 11])
        container_id = vis_dict['container_id'][img_idx].cpu().detach().numpy()
        plt.plot(container_id)
        plt.grid()
        cover_similarity = 1 - spatial.distance.cosine(cover_id, container_id)
        fused_similarity = 1 - spatial.distance.cosine(input_feature + 1e-5, container_id)
        plt.title('Container_id CovSim:{:.2f} FusSim:{:.2f}'.format(cover_similarity, fused_similarity))

        fig.add_subplot(gs[img_idx, 12])
        secret_feature_output = vis_dict['secret_feature_output'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_output)
        plt.grid()
        featinput_similarity = 1 - spatial.distance.cosine(secret_feature_input + 1e-5, secret_feature_output + 1e-5)
        plt.title('Secret_Feature_Output FeatInputSim:{:.2f}'.format(featinput_similarity))

        fig.add_subplot(gs[img_idx, 13])
        secret_feature_rec = vis_dict['secret_feature_rec'][img_idx].cpu().detach().numpy()
        plt.plot(secret_feature_rec)
        plt.grid()
        featrec_similarity = 1 - spatial.distance.cosine(secret_feature_input + 1e-5, secret_feature_rec + 1e-5)
        featoutput_similarity = 1 - spatial.distance.cosine(secret_feature_output + 1e-5, secret_feature_rec + 1e-5)
        plt.title('Secret_Feature_Rec FeatRecSim:{:.2f} FeatOutputSim: {:.2f}'.format(featrec_similarity, featoutput_similarity))

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