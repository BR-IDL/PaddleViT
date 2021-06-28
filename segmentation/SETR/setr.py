import paddle
import paddle.nn as nn
import copy
import numpy as np
from utils import *
from config import *

class Embeddings(nn.Layer):
    def __init__(self, config, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = config.MODEL.TRANS.HYBRID
        image_size = (config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)

        if self.hybrid:
            #TODO: add resnet model 
            self.hybrid_model = None

        if config.MODEL.TRANS.PATCH_GRID is not None:
            self.hybrid = True
            grid_size = config.MODEL.TRANS.PATCH_GRID
            patch_size = (image_size[0] // 16 // grid_size, image_size[1] // 16 // grid_size)
            n_patches = (image_size[0] // 16) * (image_size[1] // 16)
        else:
            self.hybrid = False
            patch_size = config.MODEL.TRANS.PATCH_SIZE
            n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        self.patch_embeddings = nn.Conv2D(in_channels=in_channels,
                                          out_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.position_embeddings = paddle.create_parameter(
                                    shape=[1, n_patches+1, config.MODEL.TRANS.HIDDEN_SIZE],
                                    dtype='float32',
                                    default_initializer=paddle.nn.initializer.Constant(0))
        self.add_parameter('pos_embed', self.position_embeddings)

        self.cls_token = paddle.create_parameter(
                                    shape=[1, 1, config.MODEL.TRANS.HIDDEN_SIZE],
                                    dtype='float32',
                                    default_initializer=paddle.nn.initializer.Constant(0))
        self.add_parameter('cls_token', self.cls_token)

        self.dropout = nn.Dropout(config.MODEL.DROPOUT)

    def forward(self, x):
        cls_tokens = self.cls_token[0].expand((x.shape[0], -1, -1))
        if self.hybrid:
            x = seld.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat((cls_tokens, x), axis=1)

        embeddings = x + self.position_embeddings[0] # tensor broadcast
        # SETR 
        embeddings = embeddings[:, 1:]
        #print(x.shape)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Layer):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_heads = config.MODEL.TRANS.NUM_HEADS
        self.attn_head_size = int(config.MODEL.TRANS.HIDDEN_SIZE / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             self.all_head_size*3,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if config.MODEL.TRANS.QKV_BIAS else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             config.MODEL.TRANS.HIDDEN_SIZE,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(config.MODEL.ATTENTION_DROPOUT)
        self.proj_dropout = nn.Dropout(config.MODEL.DROPOUT)

        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        #print('attn.shape=', attn.shape)
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn_weights = attn 
        attn = self.attn_dropout(attn)

        #print('attn shape=', attn.shape)
        #print('v shape=', v.shape)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)
        #print('z shape=', z.shape)
        # reshape 
        z = self.out(z)
        z = self.proj_dropout(z)
        return z, attn_weights


class Mlp(nn.Layer):
    def __init__(self, config):
        super(Mlp, self).__init__()

        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             int(config.MODEL.TRANS.MLP_RATIO * config.MODEL.TRANS.HIDDEN_SIZE),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(config.MODEL.TRANS.MLP_RATIO * config.MODEL.TRANS.HIDDEN_SIZE),
                             config.MODEL.TRANS.HIDDEN_SIZE,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU() 
        self.dropout1 = nn.Dropout(config.MODEL.DROPOUT)
        self.dropout2 = nn.Dropout(config.MODEL.DROPOUT)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()) #default in pp: xavier
        bias_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=1e-6)) #default in pp: zero
        
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderLayer(nn.Layer):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.hidden_size = config.MODEL.TRANS.HIDDEN_SIZE
        self.attn_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.mlp_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.mlp = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x, attn = self.attn(x)
        x = x + h 

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x, attn


class Encoder(nn.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList([copy.deepcopy(EncoderLayer(config)) for _ in range(config.MODEL.TRANS.NUM_LAYERS)])
        self.encoder_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.out_idx_list = tuple(range(config.MODEL.TRANS.NUM_LAYERS))

    def forward(self, x):
        self_attn = []
        outs = []
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x)
            self_attn.append(attn)
            if layer_idx in self.out_idx_list:
                outs.append(x)
        #out = self.encoder_norm(x)
        
        return outs


class Transformer(nn.Layer):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.dropout = nn.Dropout(config.MODEL.DROPOUT)
        self.encoder = Encoder(config)

    def forward(self, x):
        embedding_out = self.embeddings(x)
        embedding_out = self.dropout(embedding_out)
        encoder_outs = self.encoder(embedding_out)
        
        return encoder_outs



class Conv_MLA(nn.Layer):
    def __init__(self, in_channels=1024, mla_channels=256):
        super(Conv_MLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2D(in_channels, mla_channels, 1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2D(in_channels, mla_channels, 1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2D(in_channels, mla_channels, 1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2D(in_channels, mla_channels, 1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())

        self.mla_p2 = nn.Sequential(nn.Conv2D(mla_channels, mla_channels, 3, padding=1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2D(mla_channels, mla_channels, 3, padding=1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2D(mla_channels, mla_channels, 3, padding=1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
        self.mla_p5 = nn.Sequential(nn.Conv2D(mla_channels, mla_channels, 3, padding=1, bias_attr=False),
                                        nn.BatchNorm2D(mla_channels),
                                        nn.ReLU())
    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        print(x.shape)
        print(n, c, h, w)
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def forward(self, res2, res3, res4, res5):
        res2 = self.to_2D(res2)
        res3 = self.to_2D(res3)
        res4 = self.to_2D(res4)
        res5 = self.to_2D(res5)

        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_1x1 + mla_p3_1x1
        mla_p2_plus = mla_p2_1x1 + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4, mla_p5


class ViT_MLA(nn.Layer):
    def __init__(self, config):
        super(ViT_MLA, self).__init__()
        self.transformer = Transformer(config)
        self.mla = Conv_MLA(in_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                            mla_channels=config.MODEL.MLA.MLA_CHANNELS,
                            )
        self.mla_index = config.MODEL.MLA.MLA_INDEX


        self.norm_0 = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE)
        self.norm_1 = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE)
        self.norm_2 = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE)
        self.norm_3 = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE)
    def forward(self, x):
        B = x.shape[0]
        outs = self.transformer(x)

        c6 = self.norm_0(outs[self.mla_index[0]])
        c12 = self.norm_1(outs[self.mla_index[1]])
        c18= self.norm_2(outs[self.mla_index[2]])
        c24 = self.norm_3(outs[self.mla_index[3]])

        p6, p12, p18, p24 = self.mla(c6, c12, c18, c24)

        return (p6, p12, p18, p24)



def main():
    #paddle.set_device('gpu:4')
    paddle.set_device('cpu')
    config = get_config()
    dummy_img = np.random.randn(config.DATA.BATCH_SIZE,
                                3,
                                config.DATA.IMAGE_SIZE,
                                config.DATA.IMAGE_SIZE).astype('float32')
    dummy_tensor = paddle.to_tensor(dummy_img)

    vit = ViT_MLA(config)

    logits = vit(dummy_tensor)
    print(logits[0].shape)
    print(logits[1].shape)
    print(logits[2].shape)
    print(logits[3].shape)


if __name__ == "__main__":
    main()


