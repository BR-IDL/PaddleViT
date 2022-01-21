import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from transformer import build_transformer
from backbone import build_backbone
from matcher import build_matcher
from utils import nested_tensor_from_tensor_list
from segmentation import dice_loss
from segmentation import sigmoid_focal_loss
from box_ops import generalized_box_iou
from box_ops import box_cxcywh_to_xyxy
from box_ops import box_xyxy_to_cxcywh


def build_detr(config):
    """ build detr model from config"""
    # 1. build backbone with position embedding
    backbone = build_backbone(config)
    # 2. build transformer (encoders and decoders)
    transformer = build_transformer(config)
    # 3. build DETR model
    aux_loss = not config.EVAL # set true during training 
    detr = DETR(backbone=backbone,
                transformer=transformer,
                num_classes=config.MODEL.NUM_CLASSES,
                num_queries=config.MODEL.NUM_QUERIES,
                aux_loss=aux_loss)
    # 4. build matcher
    matcher = build_matcher()
    # 5. setup aux_loss
    weight_dict = {'loss_ce': 1., 'loss_bbox': 5., 'loss_giou': 2.}
    if aux_loss:
        aux_weight_dict = {}
        for i in range(config.MODEL.TRANS.NUM_DECODERS-1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # 6. build criterion
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(config.MODEL.NUM_CLASSES,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=0.1,
                             losses=losses)
    # 7. build postprocessers
    postprocessors = {'bbox': PostProcess()}
    return detr, criterion, postprocessors


class DETR(nn.Layer):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        embed_dim = transformer.dim
        w_attr_1, b_attr_1 = self.init_weights()
        self.class_embed = nn.Linear(embed_dim,
                                     num_classes+1,
                                     weight_attr=w_attr_1,
                                     bias_attr=b_attr_1)
        # MLP is different from transformer Mlp
        self.bbox_embed = MLP(in_dim=embed_dim,
                              hidden_dim=embed_dim,
                              out_dim=4,
                              num_layers=3)
        w_attr_2, _ = self.init_weights()
        self.query_embed = nn.Embedding(num_queries,
                                        embed_dim,
                                        weight_attr=w_attr_2)
        # project resnet output channels to transformer embed dim
        self.input_proj = nn.Conv2D(backbone.num_channels, embed_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def init_weights(self):
        w_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        b_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return w_attr, b_attr

    def _set_aux_loss(self, output_class, output_coord):
        return [{'pred_logits': a,
                 'pred_boxes': b} for a, b in zip(output_class[:-1], output_coord[:-1])]

    def forward(self, x):
        features, pos = self.backbone(x) # resnet + position embedding
        encoder_input, encoder_input_mask = features[-1].decompose() # resnet output features
        encoder_input = self.input_proj(encoder_input) # proj resnet out channals to embed_dim
        # transformer
        decoder_out, encoder_out = self.transformer(x=encoder_input,
                                                    mask=encoder_input_mask,
                                                    query_embed=self.query_embed.weight,
                                                    pos_embed=pos[-1])
        # out class and out bbox
        # decoder_out : [n_layers, batch, n_queries, dim]
        output_class = self.class_embed(decoder_out)
        output_coord = F.sigmoid(self.bbox_embed(decoder_out))
        
        # for validation: only use last layer output
        out = {'pred_logits': output_class[-1],
               'pred_boxes': output_coord[-1]}
        # for training: use intermediate outputs for loss

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_coord)

        return out

     
class SetCriterion(nn.Layer):
    """ build criterions for DETR"""
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses # [str, str, str]
        empty_w = paddle.ones([self.num_classes + 1])
        empty_w[-1] = self.eos_coef
        self.register_buffer(name='empty_weight', tensor=empty_w)

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """ Classification loss (NLL)
            Target dicts must contain 'labels', a tensor of dim [num_target_boxes]
        """
        assert 'pred_logits' in outputs
        # logits : [n_layers, batch, n_queries, n_classes]
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t['labels'].index_select(J) for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(src_logits.shape[:2], self.num_classes, dtype='int64')

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(input=src_logits, label=target_classes, weight=self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    @paddle.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the absolute error in the sum of predicted non-empty boxes
            This is not a real loss, for logging only, so no grad is set"""
        pred_logits = outputs['pred_logits'] 
        target_len = paddle.to_tensor([len(v['labels']) for v in targets])

        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1)
        card_pred = card_pred.astype('float32').sum(1)
        card_err = F.l1_loss(card_pred.astype('float32'), target_len.astype('float32'))
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ Compute bbox loss, the L1 loss and GIoU loss.
            Targets must contain 'boxes', a tensor of dim [num_target_boxes, 4]
            Target boxes are in format cx,cy,w,h, normalized by image size
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        #src_boxes = paddle.zeros([idx[0].shape[0], 4]) # this is wrong in training
        #for i in range(idx[0].shape[0]):
        #    src_boxes[i, :] = outputs['pred_boxes'][idx[0][i], idx[1][i], :]
        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = paddle.concat(
            (t['boxes'].index_select(i) for t, (_, i) in zip(targets, indices)))
        # L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # GIoU loss
        loss_giou = 1 - paddle.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                        box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.concat([paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """ Compute the mask loss, the focal loss and the dice loss
            Target must contain 'masks', a tensor of dim [num_target_boxes, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]

        valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]# upsample
    
        src_masks = F.interpolate(src_masks.unsqueeze(-1), size=target_masks.shape[-2:],
                                  mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
    
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.reshape(src_masks.shape)
        losses = {
            'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            'loss_dice': dice_loss(src_masks, target_masks, num_boxes)
            }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
            }
        assert loss in loss_map, f"loss {loss} not in loss_map"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets) # list of index pairs

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = paddle.to_tensor([num_boxes], dtype='float32')
        if dist.get_world_size() > 1:
            dist.all_reduce(num_boxes)
        num_boxes = paddle.clip(num_boxes / dist.get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses: # self.losses is a list of strings contains loss names
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, ignore then
                        continue 
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Layer):
    """ This layer converts the model's output into the format of coco api"""
    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1) # [batch, num_queries, num_classes]
        scores = prob[:, :, :-1].max(-1)
        labels = prob[:, :, :-1].argmax(-1)

        boxes = box_cxcywh_to_xyxy(out_bbox) # [x0, y0, x1, y1]
        img_h, img_w = target_sizes.unbind(1)
        scale_factor = paddle.stack([img_w, img_h, img_w, img_h], axis=1)
        scale_factor = scale_factor.unsqueeze(1)
        boxes = boxes * scale_factor

        result = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return result


class MLP(nn.Layer):
    """ Build mlp layers

    Multiple linear layers with ReLU activation(except last linear) applied.

    Args:
        in_dim:  input feature dim for mlp
        hidden_dim: input and output dims for middle linear layers
        out_dim: output dim for last linear layer
        num_layers: num of linear layers
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers -1)
        layer_list = []
        for idim, odim in zip([in_dim] + hidden_dims, hidden_dims + [out_dim]):
            w_attr, b_attr= self.init_weights()
            layer_list.append(
                nn.Linear(idim, odim, weight_attr=w_attr, bias_attr=b_attr))
        self.layers = nn.LayerList(layer_list)
        self.relu = nn.ReLU()

    def init_weights(self):
        w_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        b_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return w_attr, b_attr

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # last layer no activation
            if idx < len(self.layers) - 1:
                x = self.relu(x)
        return x
