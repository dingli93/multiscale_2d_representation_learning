import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import ipdb


def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    # ipdb.set_trace()
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    # print(loss)
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob

def bce_qp_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0


    # ipdb.set_trace()
    # print (joint_prob.shape, target_prob.shape)
    batch_size, dim_2, N_clips, _ = joint_prob.shape

    target_prob_qp = joint_prob.detach().view(batch_size, dim_2, -1)
    # softmax = nn.Softmax(dim=2)
    # target_prob_qp = F.softmax(target_prob_qp, dim=2)
    # target_prob_qp = torch.pow(target_prob_qp, 2) / (torch.sum(torch.pow(target_prob_qp,2), dim=2).expand(batch_size, dim_2, N_clips*N_clips))
    print (target_prob)
    print (target_prob_qp)

    # sorted, indices = torch.sort(target_prob, dim=2)
    pos_num = 0.01 * N_clips*N_clips
    topk, indices = torch.topk(target_prob_qp, k=math.ceil(pos_num), dim=2)
    print(topk)
    target_prob_qp[:, :, indices[:, :, pos_num]] = 1


    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    # loss = F.binary_cross_entropy(joint_prob, target_prob_qp.view(batch_size, dim_2, N_clips, -1), reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def reconstruction_loss(bcaption, target_var, textual_input, cfg):
    """
    logits: shape of (N, seq_len, vocab_size)
    target: shape of (N, seq_len)
    mask: shape of (N, seq_len)
    """
    # loss_fn = nn.NLLLoss(reduce=False)
    # # truncate to the same size
    # batch_size = logits.shape[0]
    # target = target[:, :logits.shape[1]]
    # mask = mask[:, :logits.shape[1]]
    # logits = to_contiguous(logits).view(-1, logits.shape[2])
    # target = to_contiguous(target).view(-1)
    # mask = to_contiguous(mask).view(-1)
    # loss = loss_fn(logits, target)
    # # print(loss, mask)
    # # loss = loss.float()
    # mask = mask.float()
    # output = torch.sum(loss * mask) / batch_size
    # return output
    # ipdb.set_trace()

    # ipdb.set_trace()
    if cfg.multi_scale_2d_map:
        all_scale_bcaption = torch.cat(bcaption, dim=1)
        all_scale_target = torch.cat(target_var, dim=0)
        bcaption = all_scale_bcaption
        target_var = all_scale_target

    batch_size, length, dim_word = textual_input.shape
    batch_size, k, _, _ = bcaption.shape
    bcaption = bcaption.view(-1, length - 1, dim_word)

    # target = target_var
    # text_temp = (textual_input * textual_mask).unsqueeze(1).expand(batch_size, N_samples * N_samples, length, dim_word)
    # # ipdb.set_trace()
    # target = text_temp.reshape(-1, length, dim_word)
    # target = text_temp.view(-1, length, dim_word)

    # loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    # r_loss = loss_fn(sigmoid(bcaption.reshape(-1, dim_word)), sigmoid(target[:, :-1, :].reshape(-1, dim_word)))
    # print(r_loss)
    # ipdb.set_trace()
    r_loss = torch.bmm(bcaption, target_var[:, :-1, :].transpose(1, 2))
    r_loss = sigmoid(r_loss)
    output = torch.sum(r_loss) / (k * math.pow(length-1, 2))
    return output, r_loss

def bce_cap_guide_loss(scores, masks, reconstruction_loss, textual_input, indices, cfg):
    # indices means the selected proposal on the 2d temporal map
    batch_size, length, dim_word = textual_input.shape
    # min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    if not cfg.multi_scale_2d_map:
        joint_prob = torch.sigmoid(scores) * masks
        # batch_vid = [joint_prob.reshape(batch_size, dim_vid, N_map * N_map)[i, :, indices[i, :, ]] for i in
        #              range(batch_size)]
        # vid_var = torch.stack(batch_vid)
        batch_size, _, N_map, _ = joint_prob.shape
        # print(joint_prob.shape, indices.shape)
        batch_vid = [joint_prob.squeeze().reshape(batch_size, N_map * N_map)[i, indices[i, 0, ]] for i in
                     range(batch_size)]
        vid_var = torch.stack(batch_vid)
    else:
        vid_var_list = []
        for scale_idx, score in enumerate(scores):
            joint_prob = torch.sigmoid(score) * masks[scale_idx]
            # batch_vid = [joint_prob.reshape(batch_size, dim_vid, N_map * N_map)[i, :, indices[i, :, ]] for i in
            #              range(batch_size)]
            # vid_var = torch.stack(batch_vid)
            batch_size, _, N_map, _ = joint_prob.shape
            # print(joint_prob.shape, indices.shape)
            batch_vid = [joint_prob.squeeze().reshape(batch_size, N_map * N_map)[i, indices[scale_idx][i, 0,]] for i in
                         range(batch_size)]
            vid_var = torch.stack(batch_vid)
            vid_var_list.append(vid_var)
        vid_var = torch.cat(vid_var_list, dim=1)
    # print('vid_var.shape', vid_var.shape)
    # target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    # torch.argmin(reconstruction_loss, dim=)
    # print (reconstruction_loss, reconstruction_loss.shape)
    # ipdb.set_trace()
    target_prob = F.normalize(torch.sum(reconstruction_loss, (1,2)).reshape(batch_size, -1))
    p_label = target_prob.data
    # sorted, indices = torch.sort(p_label)
    # print (p_label)
    # print (p_label.shape)

    # target_prob[target_prob > 0] += bias
    # target_prob[target_prob > 1] = 1
    # p_label[p_label > 1] = 1
    p_label[p_label < cfg.POS_THRESH] = 0
    p_label[p_label >= cfg.NEG_THRESH] = 1
    final_label = 1 - p_label
    # target_prob = 1 - target_prob.reshape(batch_size, 1, N_samples, N_samples)

    # print(scores.shape)
    # ipdb.set_trace()
    loss = F.binary_cross_entropy(vid_var, final_label, reduction='none')
    loss_value = torch.sum(loss)
    return loss_value, scores

