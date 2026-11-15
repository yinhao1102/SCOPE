"""Similarity Distribution Matching used with the SCOPE encoders."""
import torch
from torch.nn import functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-08):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    if image_id != None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    labels_distribute = labels / labels.sum(dim=1)
    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return loss
