import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

from . import make_backbone, make_generator, make_neck

from .models import register_meta_arch
from .blocks import LayerNorm, MaskedConv1D, Scale
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from libs.utils.nms import batched_nms
# zeroshot
from .text import TextFeatures



class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))

            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
                
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            # cur_offsets = torch.clamp(cur_offsets, min=-50, max=50)   # 추가
            out_offsets += (F.softplus(self.scale[l](cur_offsets)), )
            # out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 
            

        # fpn_masks remains the same
        return out_offsets

@register_meta_arch("TiFAD")
class TiFAD(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        split_num,             # file number of text data split  (0~9)
        subset_file,           # data split path
        data_split,            # 50 for 50%-50% setting and 75 for 75%-25% setting
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        text_prior_prob,       # text prior probability
        topk_ratio,            # topk ratio of salient attentive mask 
        threshold,             # threshold of salient attentive mask 
        text_model,            # text model (e.g., CLIP)
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
    ):
        super().__init__()

        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']
        
        # backbone
        self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim, 
                    'n_head': n_head,    
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,    
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,          
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'topk_ratio' : topk_ratio,
                    'threshold' : threshold,
                }
            )
        
        # fpn network: convs
        # assert fpn_type in ['fpn', 'identity']
        # self.neck = make_neck(
        #     fpn_type,
        #     **{
        #         'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
        #         'out_channel' : fpn_dim,
        #         'scale_factor' : scale_factor,
        #         'start_level' : fpn_start_level,
        #         'with_ln' : fpn_with_ln
        #     }
        # )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # regerssion head
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
        )

        # foreground head
        self.fg_head = PtTransformerClsHead(
            fpn_dim, head_dim, 1,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob, 
            with_ln=head_with_ln,
            num_layers=head_num_layers,   
            empty_cls=train_cfg['head_empty_cls']
        )

        # centerness head
        self.cn_head = PtTransformerClsHead(
            fpn_dim, head_dim, 1,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob, 
            with_ln=head_with_ln,
            num_layers=head_num_layers,   
            empty_cls=train_cfg['head_empty_cls']
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        # text setting
        self.text_model = text_model
        self.text_proj = nn.Linear(embd_dim, embd_dim)
        self.bias_lang = nn.Parameter(torch.zeros(embd_dim), requires_grad=True)
        bias_value = -math.log((1 - text_prior_prob) / text_prior_prob)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.zeros([]))
        self.subset_file = subset_file
        self.data_split = data_split

        self.text_features = TextFeatures(
            model_path=self.text_model,
            subset_file=self.subset_file,
            data_split=self.data_split,
            emb_dim=embd_dim,
            split_num=split_num,
        )
    
        
    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        batch_size, _, _ = batched_inputs.shape

        # text embedding
        mode ='train' if self.training else 'test'
        text_embed, new_cls_num = self.text_features(batch_size, mode)

        updated_feat, fpn_masks, updated_text = self.backbone(batched_inputs, text_embed, batched_masks) 
        updated_feat = [x.transpose(2, 1) for x in updated_feat]

        updated_text = updated_text[:, :new_cls_num]

        updated_text_lv = []
        for l in range(len(updated_feat)):
            updated_text_lv.append(updated_text)

        updated_feat = [feat.permute(0, 2, 1) for feat in updated_feat]

        # foreground (out_fg)
        out_fg = self.fg_head(updated_feat, fpn_masks)       
        out_fg = [x.permute(0, 2, 1) for x in out_fg]

        # regressgion (out_offsets)
        out_offsets = self.reg_head(updated_feat, fpn_masks)      
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # centerness (out_centerness)
        out_centerness = self.cn_head(updated_feat, fpn_masks)
        out_centerness = [x.permute(0, 2, 1) for x in out_centerness]

        # classification (out_cls_logits)
        updated_text_lv = [text / text.norm(dim=-1, keepdim=True) for text in updated_text_lv]
        updated_text_bias_lv = [torch.matmul(text, self.bias_lang) + self.bias0 for text in updated_text_lv]        
        updated_text_lv = [self.text_proj(text / 2.0) for text in updated_text_lv]
        for i, text in enumerate(updated_text_lv):
            if torch.isnan(text).any():
                print(f"updated_text_lv[{i}] contains NaN!")

        updated_feat_text_matmul = [
            (feat.transpose(2, 1) @ text.transpose(2, 1) / self.logit_scale.exp()) + bias.unsqueeze(1).repeat(1, feat.size(-1), 1)
            for feat, text, bias in zip(updated_feat, updated_text_lv, updated_text_bias_lv)
        ]
        
        out_cls_logits = updated_feat_text_matmul
        out_cls_logits = [torch.clamp(logit, min=-50000, max=50000) for logit in out_cls_logits]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks] 

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(updated_feat)     
    
        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, gt_fg, gt_cn = self.label_points(
                points, gt_segments, gt_labels, new_cls_num)   

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets, 
                gt_cls_labels, gt_offsets, new_cls_num,   
                out_fg, gt_fg, 
                out_centerness, gt_cn,
            )

            return losses

        else:
            # torch.sum(out_cls_logits[0].sigmoid(), 1)
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, out_fg, out_centerness, 
            )

            return results


    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])   
        max_len = feats_lens.max(0).values.item()      

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len  
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len] 
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, new_cls_num):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset, gt_fg, gt_cn = [], [], [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets, fg_targets, cn_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label, new_cls_num
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            gt_fg.append(fg_targets)
            gt_cn.append(cn_targets)

        return gt_cls, gt_offset, gt_fg, gt_cn

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, new_cls_num):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]     
        num_gts = gt_segment.shape[0]        

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, new_cls_num), 0)   
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            fg_target = torch.zeros(num_pts, device=gt_label.device)   
            cn_targets = torch.zeros(num_pts, device=gt_label.device)   
            return cls_targets, reg_targets, fg_target, cn_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)  
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]      
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]     
        reg_targets = torch.stack((left, right), dim=-1)


        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])  
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2; video_target: C
        gt_label_one_hot = F.one_hot(
            gt_label, new_cls_num 
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]
        
        # foreground target
        fg_target = cls_targets.sum(dim=-1)
        fg_target.clamp_(min=0.0, max=1.0)
        fg_target = fg_target.to(reg_targets.device)

        # centerness target
        mask = reg_targets.min(-1)[0] > 0
        sigma = 1.0
        gaussian_cn_targets = torch.exp(-0.5 * ((reg_targets / sigma) ** 2))
        gaussian_cn_values, _ = gaussian_cn_targets.min(dim=-1)
        gaussian_cn_targets = gaussian_cn_values * mask.float()

        return cls_targets, reg_targets, fg_target, gaussian_cn_targets
                
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, new_cls_num,
        out_fg, gt_fg,
        out_centerness, gt_cn,
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)    
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)
        self.loss_normalizer2 = max(num_pos, 1)  

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (new_cls_num + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # 3. foreground loss
        pred_fg = torch.cat(out_fg, dim=1).squeeze(-1)
        fg_loss = sigmoid_focal_loss(
            pred_fg[valid_mask],
            torch.stack(gt_fg)[valid_mask],
            reduction='sum'
        )
        fg_loss /= self.loss_normalizer 

        # 4. centerness loss 
        cn_loss = sigmoid_focal_loss(
            torch.cat(out_centerness, dim=1)[valid_mask].squeeze(-1),
            torch.stack(gt_cn)[valid_mask],
            reduction='sum' 
        )
        cn_loss /= self.loss_normalizer

        # return a dict of losses
        final_loss = 0.5 * cls_loss + reg_loss +  fg_loss +  cn_loss
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'fg_loss' : fg_loss,
                'cn_loss' : cn_loss,
                'final_loss' : final_loss}

    
    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets, 
        out_fg,
        out_centerness,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            fg_per_vid = [x[idx] for x in out_fg]
            cn_per_vid = [x[idx].sigmoid() for x in out_centerness]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, fg_per_vid, cn_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results_per_vid['class_names'] = self.text_features.test_classes 
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets, 
        out_fg,
        out_cn
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        pts_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i, fg_i, cn_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks, out_fg, out_cn):

            # sigmoid normalization for output logits
            fg_i = (fg_i.sigmoid() * mask_i.unsqueeze(-1))
            cn_i = (cn_i.sigmoid() * mask_i.unsqueeze(-1))

            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1))
            pred_prob = torch.sqrt(pred_prob * cn_i)
            pred_prob = pred_prob * fg_i
            pred_prob = pred_prob.flatten()
            
            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_cls = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_cls.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_cls = topk_cls[idxs[:num_topk]].clone()


            # fix a warning in pytorch 1.9
            num_cls = cls_i.size(-1)
            pt_idxs =  torch.div(          
                topk_cls, num_cls, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_cls, num_cls)
            
            
            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]  

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)
            

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])
            pts_idxs_all.append(pts[:, 0][keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all, pts_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all, pts_idxs_all]
        ]
        
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all,
                   }
        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps 
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            class_names = results_per_vid['class_names']

            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu() 
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                # segs, scores, labels, temp_idxs = batched_nms(
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )

            # 3: convert from feature grids to seconds  (real time)
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen

            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels,
                 'class_names': class_names,
                 }
            )
        return processed_results
