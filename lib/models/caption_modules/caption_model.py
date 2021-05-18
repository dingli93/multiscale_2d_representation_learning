import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import ipdb
import numpy as np

class MyCaptionModel(nn.Module):
    # def __init__(self, max_len, dim_hidden, dim_word, cfg, dim_vid=2048, sos_id=1, eos_id=0,
    #              n_layers=1, bidirectional=False, rnn_cell='lstm', rnn_dropout_p=0.2):
    def __init__(self, cfg):
        # python 3
        # super().__init__()
        super(MyCaptionModel, self).__init__()
        rnn_cell = cfg.RNN_CELL
        dim_hidden = cfg.DIM_HIDDEN
        dim_word = cfg.DIM_WORD
        dim_vid = cfg.DIM_VID
        n_layers = cfg.N_LAYERS
        self.topk = cfg.TOPK

        bidirectional = False
        rnn_dropout_p = 0.2
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        #  hidden_size * num_directions
        #  num_directions = 2 if bidirectional else 1
        rnn_output_size = dim_hidden * 2 if bidirectional else dim_hidden

        self.rnn1 = self.rnn_cell(dim_word, dim_hidden, n_layers, bidirectional=bidirectional,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(rnn_output_size + dim_vid, dim_hidden, n_layers, bidirectional=bidirectional,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn_cell_type = rnn_cell.lower()
        self.n_layers = n_layers
        self.dim_vid = dim_vid
        # self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        # self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        # self.embedding = nn.Embedding(self.dim_output, self.dim_word)


        self.out = nn.Linear(self.dim_hidden, self.dim_word)
        self.max_text = nn.MaxPool1d(3, stride=2)



        # self.init_weights()

    # def init_weights(self):
    #     self.embedding.weight.data.uniform_(-0.1, 0.1)

    def sample_topk_proposals(self, prediction, masks):
        joint_prob = torch.sigmoid(prediction) * masks
        batch_size, _, N_clips, N_clips = joint_prob.shape
        topk_prob, indices = torch.topk(joint_prob.view(batch_size, 1, -1), k=10)
        return indices

    def forward(self, vid_feats, map_mask, prediction, masks, target_variable=None, target_variable_mask=None,
                mode='train', opt={}):

        batch_size, n_words, dim_word = target_variable.shape
        batch_size, dim_vid, N_map, _ = vid_feats.shape
        # out_caption = Variable(vid_feats.data.new(batch_size, N_map, N_map, n_words - 1 , self.dim_word)).zero_()
        # if torch.cuda.is_available():
        #     out_caption = out_caption.cuda()
        # padding_words = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
        # if torch.cuda.is_available():
        #     padding_words = padding_words.cuda()
        state1 = None
        state2 = None
        # self.rnn1.flatten_parameters()
        # self.rnn2.flatten_parameters()
        # max_pool_1d = torch.nn.MaxPool1d(vid_feats.shape[1])
        # vid_feat_temp = max_pool_1d(torch.transpose(vid_feats, 1, 2))
        # vid_pool_feats = torch.transpose(vid_feat_temp, 1, 2)

        # output1, state1 = self.rnn1(target_variable, state1)
        # input2 = torch.cat((output1, padding_words), dim=2)
        # output2, state2 = self.rnn2(input2, state2)

        # padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_word)).zero_()
        # if torch.cuda.is_available():()
        #     padding_frames = padding_frames.cuda

        if mode == 'train':
            joint_prob = torch.sigmoid(prediction) * masks
            topk_prob, indices = torch.topk(joint_prob.view(batch_size, 1, -1), k=self.topk)

            batch_vid = [(map_mask * vid_feats).reshape(batch_size, dim_vid, N_map*N_map)[i, :, indices[i, :, ]] for i in range(batch_size)]
            vid_var = torch.stack(batch_vid)
            # print('vid_var', vid_var.shape)
            input_vid = vid_var.transpose(1, 3).reshape(-1, dim_vid).unsqueeze(1).expand(-1, n_words-1, dim_vid)
            # print(target_variable)
            target_variable = target_variable_mask * target_variable
            target_variable = target_variable.unsqueeze(1).expand(batch_size, self.topk, n_words, dim_word)
            # target_variable = target_variable.unsqueeze(1).expand(batch_size, N_map, N_map, n_words, dim_word)
            target_var = target_variable.reshape(-1, n_words, dim_word)

            self.rnn1.flatten_parameters()
            self.rnn2.flatten_parameters()
            output1, state1 = self.rnn1(target_var[:, :-1, :])
            input2 = torch.cat((input_vid, output1), dim=2)
            output2, state2 = self.rnn2(input2)
            logits = self.out(output2.reshape(-1, self.dim_hidden))
            logits = F.log_softmax(logits, dim=1)
            logits = logits.view(-1, n_words-1, dim_word)


            out_caption = logits.view(batch_size, self.topk, n_words-1, self.dim_word)
            # seq_preds = []

            # # for i in range(self.max_length - 1):
            # for m in range(2):
            #     for n in range(2):
            #         # print(map_mask[:, :, m, n].data.cpu())
            #         # print(torch.zeros(batch_size, 1).cpu())
            #         # print((map_mask[:, :, m, n].data.cpu() == torch.zeros(batch_size, 1).cpu())[0, 0].item())
            #         if (map_mask[:, :, m, n].data.cpu() == torch.zeros(batch_size, 1).cpu())[0, 0].item():
            #             continue
            #
            #         seq_preds = []
            #         words = []
            #         for i in range(target_variable.shape[1] - 1):
            #             # <eos> doesn't input to the network
            #             if i == 1 or i == 3:
            #                 current_words = target_variable[:, i, :].view(batch_size, 1, -1).zero_() - 1
            #             else:
            #                 current_words = target_variable[:, i, :].view(batch_size, 1, -1)
            #             # self.rnn1.flatten_parameters()
            #             # self.rnn2.flatten_parameters()
            #             output1, state1 = self.rnn1(current_words, state1)
            #
            #             input2 = torch.cat(
            #                 (vid_feats[:, :, m, n].unsqueeze(1), output1), dim=2)
            #             output2, state2 = self.rnn2(input2, state2)
            #
            #             logits = self.out(output2.squeeze(1))
            #             logits = F.log_softmax(logits, dim=1)
            #             words.append(logits.unsqueeze(1))
            #             # seq_probs.append(logits.unsqueeze(1))
            #         # seq_probs = torch.cat(seq_probs, 1)
            #         seq_probs = torch.cat(words, 1)
            #         out_caption[:, m, n, :, :] = seq_probs
            # print('Here')
            # seq_words = torch.cat(words, 1)
        else:
            beam_size = opt.get('beam_size', 1)
            if beam_size == 1:
                current_words = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
                for i in range(self.max_length - 1):
                    self.rnn1.flatten_parameters()
                    self.rnn2.flatten_parameters()
                    output1, state1 = self.rnn1(padding_frames, state1)
                    input2 = torch.cat(
                        (output1, current_words.unsqueeze(1)), dim=2)
                    output2, state2 = self.rnn2(input2, state2)
                    logits = self.out(output2.squeeze(1))
                    logits = F.log_softmax(logits, dim=1)
                    seq_probs.append(logits.unsqueeze(1))
                    _, preds = torch.max(logits, 1)
                    current_words = self.embedding(preds)
                    seq_preds.append(preds.unsqueeze(1))
                seq_probs = torch.cat(seq_probs, 1)
                seq_preds = torch.cat(seq_preds, 1)
            else:
                # batch*dim_word
                start = [Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()]
                current_words = [[start, 0.0, state2]]
                for i in range(self.max_length - 1):
                    self.rnn1.flatten_parameters()
                    self.rnn2.flatten_parameters()
                    # output1: batch*1*dim_hidden
                    output1, state1 = self.rnn1(padding_frames, state1)
                    temp = []
                    for s in current_words:
                        # s: [[batch*word_embed1, batch*word_embed2...], prob, state2]
                        input2 = torch.cat(
                            (output1, self.embedding(s[0][-1]).unsqueeze(1)), dim=2)
                        output2, s[2] = self.rnn2(input2, s[2])
                        logits = self.out(output2.squeeze(1))
                        # batch*voc_size
                        logits = F.log_softmax(logits, dim=1)
                        # batch*beam
                        topk_prob, topk_word = torch.topk(logits, k=beam_size, dim=1)
                        # batch*beam -> beam*batch
                        topk_prob = topk_prob.permute(1, 0)
                        topk_word = topk_word.permute(1, 0)
                        # Getting the top <beam_size>(n) predictions and creating a
                        # new list so as to put them via the model again
                        for prob, word in zip(topk_prob, topk_word):
                            next_cap = s[0][:]
                            next_cap.append(word)
                            temp.append([next_cap, s[1]+prob,
                                         (s[2][0].clone(), s[2][1].clone()) if isinstance(s[2], tuple)
                                         else s[2].clone()])
                    current_words = temp
                    # sort by prob
                    current_words = sorted(current_words, reverse=False, cmp=lambda x,y:cmp(int(x[1]),int(y[1])))
                    # get the top words
                    current_words = current_words[-beam_size:]
                seq_preds = torch.cat(current_words[-1][0][1:], 0).unsqueeze(0)
        return out_caption, target_var, indices