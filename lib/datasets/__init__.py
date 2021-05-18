import torch
import torch.nn as nn
from core.config import config
import ipdb

def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    # ipdb.set_trace()

    if not config.TRAIN.multi_scale_2d_map_for_full_sup:
        max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
        padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
        for i, map_gt in enumerate(batch_map_gt):
            num_clips = map_gt.shape[-1]
            padded_batch_map_gt[i][0,:num_clips,:num_clips] = map_gt

        batch_data = {
            'batch_anno_idxs': batch_anno_idxs,
            'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
            'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
            'batch_map_gt': padded_batch_map_gt,
            'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
            'batch_duration': batch_duration,
        }
    else:
        # 2nd and 3rd, 4th, ... max num_clips doesn't need padding
        batch_map_gt_list = []
        max_num_clips = max([map_gt[0].shape[-1] for map_gt in batch_map_gt])
        padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
        for j in range(len(config.TRAIN.num_clips_list)):
            for i, map_gt in enumerate(batch_map_gt):
                num_clips = map_gt[0].shape[-1]
                if j == 0:
                    padded_batch_map_gt[i][0, :num_clips, :num_clips] = map_gt[0]
                    map_gt[0] = padded_batch_map_gt[i]
                else:
                    max_num_clips = config.TRAIN.num_clips_list[j]
                    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
            batch_map_gt_list.append(padded_batch_map_gt)

        # ipdb.set_trace()
        batch_data = {
            'batch_anno_idxs': batch_anno_idxs,
            'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
            'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
            'batch_map_gt': batch_map_gt_list,
            'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
            'batch_duration': batch_duration,
        }

    return batch_data

def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

from datasets.activitynet import ActivityNet
from datasets.charades import Charades
from datasets.tacos import TACoS
