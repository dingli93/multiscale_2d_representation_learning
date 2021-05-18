from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.caption_modules as caption_modules
import ipdb

class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()
        # self.scale = config.TAN.SCALE
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        if config.TRAIN.caption_eval_prop:
            self.bcap_layer = getattr(caption_modules, config.TAN.CAPTION_MODULE.NAME)(config.TAN.CAPTION_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, visual_input):

        if not config.TRAIN.multi_scale_2d_map:
            # print('visual_input: ', visual_input.shape)  # [B, 256, 4096]
            vis_h = self.frame_layer(visual_input.transpose(1, 2))
            # print('vis_h: ', vis_h.shape)  # [B, DIM_FEAT, N]

            map_h, map_mask = self.prop_layer(vis_h)
            # print('map_mask: ', map_mask.shape)  # [B, 1, N, N]
            # print('map_h: ', map_h.shape)  # [B, DIM_FEAT, N, N]

            #shape fused_h: [B, DIM_FEAT, N, N]   N is width of 2d temporal map
            fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
            # print('fused_h: ', fused_h.shape)
            fused_h = self.map_layer(fused_h, map_mask)
            # print('fused_h: ', fused_h.shape)
            prediction = self.pred_layer(fused_h) * map_mask
            # print('prediction: ', prediction.shape)  # [B, 1, N, N]

            # ipdb.set_trace()
            if config.TRAIN.caption_eval_prop:
                back_caption, target_var, indices = self.bcap_layer(fused_h, map_mask, prediction, map_mask,
                                                                    textual_input, textual_mask)
                return back_caption, target_var, indices, prediction, map_mask
            else:
                return prediction, map_mask
        else:
            prediction_list = []
            back_caption_list = []
            target_var_list = []
            indices_list = []

            # print('visual_input: ', visual_input.shape)  # [B, 256, 4096]
            vis_h = self.frame_layer(visual_input.transpose(1, 2))
            # print('vis_h: ', vis_h.shape)  # [B, DIM_FEAT, N]
            map_h_list, map_mask_list = self.prop_layer(vis_h)
            # print('map_mask: ', map_mask.shape)  # [B, 1, N, N]
            # print('map_h: ', map_h.shape)  # [B, DIM_FEAT, N, N]
            # ipdb.set_trace()

            # shape fused_h: [B, DIM_FEAT, N, N]   N is width of 2d temporal map

            for i, map_h in enumerate(map_h_list):
                fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask_list[i])
                # print('fused_h: ', fused_h.shape)
                fused_h = self.map_layer(fused_h, map_mask_list[i])
                # print('fused_h: ', fused_h.shape)
                prediction = self.pred_layer(fused_h) * map_mask_list[i]
                # print('prediction: ', prediction.shape)  # [B, 1, N, N]
                prediction_list.append(prediction)

                # ipdb.set_trace()
                if config.TRAIN.caption_eval_prop:
                    back_caption, target_var, indices= self.bcap_layer(fused_h, map_mask_list[i], prediction, map_mask_list[i], textual_input, textual_mask)
                    back_caption_list.append(back_caption)
                    target_var_list.append(target_var)
                    indices_list.append(indices)
                    # return back_caption, target_var, indices, prediction, map_mask_list
            # ipdb.set_trace()
            if config.TRAIN.caption_eval_prop:
                return back_caption_list, target_var_list, indices_list, prediction_list, map_mask_list
            else:
                return prediction_list, map_mask_list

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
