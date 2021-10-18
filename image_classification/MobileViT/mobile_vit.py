from einops import rearrange
import paddle
import paddle.nn as nn


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, oup, kernal_size, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )

class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        
        q, k, v = (paddle.to_tensor(i,stop_gradient=False) 
                        for i in map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = 1), 
                        (i.numpy() for i in qkv)))
        dots = paddle.matmul(q, k.transpose((0,1,2,4,3))) * self.scale
        attn = self.attend(dots)
        out = paddle.matmul(attn, v)
        out = paddle.to_tensor(rearrange(out.numpy(), 'b p h n d -> b p n (h d)'),stop_gradient=False)
        return self.to_out(out)

class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Layer):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                # dw
                nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileViTBlock(nn.Layer):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

         # Global representations
        _, _, h, w = x.shape
        x = paddle.to_tensor(rearrange(x.numpy(), 'b d (h ph) (w pw) -> b (ph pw) (h w) d', 
                                ph=self.ph, pw=self.pw),
                                stop_gradient=False)
        x = self.transformer(x)
        x = paddle.to_tensor(rearrange(x.numpy(), 'b (ph pw) (h w) d -> b d (h ph) (w pw)', 
                                h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw),
                                stop_gradient=False)

        # Fusion
        x = self.conv3(x)
        x = paddle.concat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Layer):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.LayerList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.LayerList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2D(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias_attr=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).reshape([-1, x.shape[1]])
        x = self.fc(x)
        return x


def build_mobile_vit(config):
    """Build mlp mixer by reading options in config object
    Args:
        config: config instance contains setting options
    Returns:
        model: MlpMixer model
    """
    dims = config.MODEL.DIMS
    channels = config.MODEL.CHANNELS
    num_classes = config.MODEL.NUM_CLASSES
    expaansion = config.MODEL.MV2_EXPAANSION
    model = MobileViT((256,256),dims,channels,num_classes,expaansion)
    return model


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


if __name__ == '__main__':
    img = paddle.randn([5, 3, 256, 256])

    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))