import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from resnet import ResNet18
from transformer import Transformer

paddle.set_device('cpu')


class PositionEmbedding(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.row_embed = nn.Embedding(50, embed_dim)
        self.col_embed = nn.Embedding(50, embed_dim)
        
    def forward(self, x):
        # x: [b, feat, H, W]
        h, w = x.shape[-2:]
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(i)
        pos = paddle.concat([x_embed.unsqueeze(0).expand((h, x_embed.shape[0], x_embed.shape[1])),
                             y_embed.unsqueeze(1).expand((y_embed.shape[0], w, y_embed.shape[1]))], axis=-1)
        pos = pos.transpose([2, 0, 1])
        pos = pos.unsqueeze(0)
        pos = pos.expand([x.shape[0]] + pos.shape[1::]) #[batch_size, embed_dim, h, w]
        return pos


class BboxEmbed(nn.Layer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class DETR(nn.Layer):
    def __init__(self, backbone, pos_embed, transformer, num_classes, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        embed_dim = transformer.embed_dim

        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed  = BboxEmbed(embed_dim, embed_dim, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.input_proj = nn.Conv2D(backbone.num_channels, embed_dim, kernel_size=1)
        self.backbone = backbone
        self.pos_embed = pos_embed

    def forward(self, x):
        print(f'----- INPUT: {x.shape}')
        feat = self.backbone(x)
        print(f'----- Feature after ResNet18: {feat.shape}')
        pos_embed = self.pos_embed(feat)
        print(f'----- Positional Embedding: {pos_embed.shape}')
        
        feat = self.input_proj(feat)
        print(f'----- Feature after input_proj: {feat.shape}')
        out, _ = self.transformer(feat, self.query_embed.weight, pos_embed)
        print(f'----- out after transformer: {out.shape}')

        out_class = self.class_embed(out)
        out_coord = self.bbox_embed(out)
        print(f'----- out for class: {out_class.shape}')
        print(f'----- out for bbox: {out_coord.shape}')
        #out_coord = F.sigmoid(out_coord)

        return out_class, out_coord


def build_detr():
    backbone = ResNet18() 
    transformer = Transformer()
    pos_embed = PositionEmbedding(16)
    detr = DETR(backbone, pos_embed, transformer, 10, 100)
    return detr


def main():
    t = paddle.randn([3, 3, 224, 224])
    model = build_detr()
    out = model(t)
    print(out[0].shape, out[1].shape)



if __name__ == "__main__":
    main()
    

