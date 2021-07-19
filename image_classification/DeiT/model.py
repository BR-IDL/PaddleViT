"""
Implement Transformer Class for ViT
"""

import paddle
import paddle.nn as nn
from transformer import Embeddings
from transformer import Identity
from transformer import VisualTransformer

class DeitEmbeddings(Embeddings):
    """Embedding Layer for DeiT introduced from ViT.

    Embedding Layer for DeiT, add dist_token to improve the performance.

    Extra attributes:
        dist_token: token inserted to the patch after the cls_token to help distillation
    Modified attributes:
        position_embeddings: 
        considering the introduction of dist_token, 
        the length is modified from num_patch + 1 to num_patch + 2
    """
    def __init__(self, config, in_channels=3):
        super().__init__(config, in_channels=in_channels)
        #Compared with ViT, the shape of positon_embeddings is different with the introduction of distillation token 
        n_patches = self.position_embeddings.shape[1]+1
        self.position_embeddings = self.create_parameter(
            shape=(1, n_patches, config.MODEL.TRANS.HIDDEN_SIZE),
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.dist_token = self.create_parameter(
            shape=(1, 1, config.MODEL.TRANS.HIDDEN_SIZE),
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
    
    def forward(self, x):
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        dist_tokens = self.dist_token.expand((x.shape[0], -1, -1))
        if self.hybrid:
            #x = self.hybrid_model(x) #TODO
            print("hybrid model TODO")
            pass
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose((0,2,1))
        x = paddle.concat((cls_tokens, dist_tokens, x), axis=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
         



class DistilledVisionTransformer(VisualTransformer):
    """ Vision Transformer with distillation token introduced from ViT
    
    DeiT, Compared with ViT, introduce dist_token for distillation contributing to better performance.

    Extra Attributes:
        head_dist: single Linear Layer for distillation
    
    Modified Attributes:
        transformer.embeddings: 
        with the introduction of dist_token, 
        there is some difference while embedding.


    Extra attributes
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.transformer.embeddings = DeitEmbeddings(config)
        head_attr, head_bias_attr = self._init_weights()
        self.head_dist = nn.Linear(
                config.MODEL.TRANS.HIDDEN_SIZE,
                config.MODEL.NUM_CLASSES,
                weight_attr=head_attr,
                bias_attr=head_bias_attr
                ) if config.MODEL.NUM_CLASSES else Identity()

    def forward(self, x):
        x, self_attn = self.transformer(x)
        x, x_dist = x[:,0], x[:,1]
        x = self.classifier(x) 
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        # during inference, return the average of both classifier predictions
        return (x + x_dist) / 2
