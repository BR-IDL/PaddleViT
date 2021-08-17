import paddle
import numpy as np
import paddle.nn as nn
from paddle.nn import Softmax


def INF(B,H,W):
     return -paddle.diag(paddle.to_tensor(float("inf")).tile([H]),0).unsqueeze(0).tile([B*W,1,1])


class CrissCrossAttention(nn.Layer):
    """ Criss-Cross Attention Layer"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(axis=3)
        self.INF = INF
        self.gamma = paddle.create_parameter(shape=[1], dtype='float32', default_initializer=nn.initializer.Constant(0.0))


    def forward(self, x):
        m_batchsize, _, height, width = x.shape
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.transpose([0,3,1,2]).reshape([m_batchsize*width,-1,height]).transpose([0, 2, 1])
        proj_query_W = proj_query.transpose([0,2,1,3]).reshape([m_batchsize*height,-1,width]).transpose([0, 2, 1])
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.transpose([0,3,1,2]).reshape([m_batchsize*width,-1,height])
        proj_key_W = proj_key.transpose([0,2,1,3]).reshape([m_batchsize*height,-1,width])
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.transpose([0,3,1,2]).reshape([m_batchsize*width,-1,height])
        proj_value_W = proj_value.transpose([0,2,1,3]).reshape([m_batchsize*height,-1,width])
        energy_H = (paddle.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).reshape([m_batchsize,width,height,height]).transpose([0,2,1,3])
        energy_W = paddle.bmm(proj_query_W, proj_key_W).reshape([m_batchsize,height,width,width])
        concate = self.softmax(paddle.concat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].transpose([0,2,1,3]).reshape([m_batchsize*width,height,height])
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].reshape([m_batchsize*height,width,width])
        out_H = paddle.bmm(proj_value_H, att_H.transpose([0, 2, 1])).reshape([m_batchsize,width,-1,height]).transpose([0,2,3,1])
        out_W = paddle.bmm(proj_value_W, att_W.transpose([0, 2, 1])).reshape([m_batchsize,height,-1,width]).transpose([0,2,1,3])
        #print(out_H.shape,out_W.shape)
        return self.gamma*(out_H + out_W) + x



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    np.random.seed(0)
    x_numpy = np.random.random((2, 64, 5, 6)).astype('float32')
    x = paddle.to_tensor(x_numpy)
    out = model(x)
    print(out.shape)
