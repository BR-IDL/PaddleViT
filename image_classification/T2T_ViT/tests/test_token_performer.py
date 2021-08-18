import unittest
import numpy as np
import paddle
import paddle.nn as nn
import torch
from t2t_vit import *


class TokenPerformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
    
    @classmethod
    def tearDown(cls):
        pass

    def test_prm_exp_einsum(self):
        x = np.random.randn(2, 3136, 64).astype('float32') 
        w = np.random.randn(32, 64).astype('float32')
        m = 32 
        # pytorch 
        x_pth = torch.Tensor(x)
        w_pth = torch.Tensor(w)
        xd_pth = (x_pth * x_pth).sum(dim=-1, keepdim=True).repeat(1, 1, m) / 2
        #print(xd_pth)
        wtx_pth = torch.einsum('bti,mi->btm', x_pth.float(), w_pth)
        #print(wtx_pth)
        out_pth = torch.exp(wtx_pth - xd_pth) / math.sqrt(m)
        #print('-------------------')
        # paddle
        x_pd = paddle.to_tensor(x)
        w_pd = paddle.to_tensor(w)
        xd_pd = (x_pd * x_pd).sum(axis=-1, keepdim=True)
        xd_pd = xd_pd.expand([xd_pd.shape[0], xd_pd.shape[1], m]) / 2
        #print(xd_pd)
        wtx_pd = paddle.matmul(x_pd, w_pd, transpose_y=True)
        #print(wtx_pd)
        out_pd = paddle.exp(wtx_pd - xd_pd) / math.sqrt(m)

        # check if paddle out equals to pytorch out 
        out_pth_np = out_pth.cpu().numpy()
        out_pd_np = out_pd.cpu().numpy()
        self.assertTrue(np.allclose(out_pth_np, out_pd_np, atol=1e-5))

    def test_single_attention_einsum(self):
        qp = np.random.randn(2, 3136, 32).astype('float32') 
        kp = np.random.randn(2, 3136, 32).astype('float32') 
        v = np.random.randn(2, 3136, 64).astype('float32') 
        emb = 64 
        # pytorch 
        qp_pth = torch.Tensor(qp)
        kp_pth = torch.Tensor(kp)
        v_pth = torch.Tensor(v)
        D_pth = torch.einsum('bti,bi->bt', qp_pth, kp_pth.sum(dim=1)).unsqueeze(dim=2)
        #print(D_pth.shape)
        #print('D_pth: ', D_pth)
        kptv_pth = torch.einsum('bin,bim->bnm', v_pth.float(), kp_pth)
        #print(kptv_pth)
        y_pth = torch.einsum('bti,bni->btn', qp_pth, kptv_pth)
        y_pth = y_pth / (D_pth.repeat(1, 1, emb) + 1e-3)
        #print('y_pth = ', y_pth)

        #print('-------------------')
        # paddle
        qp_pd = paddle.to_tensor(qp)
        kp_pd = paddle.to_tensor(kp)
        v_pd = paddle.to_tensor(v)
        D_pd = paddle.matmul(qp_pd, kp_pd.sum(axis=1).unsqueeze(2)) 
        #print(D_pd.shape)
        #print('D_pd: ', D_pd)
        kptv_pd = paddle.matmul(v_pd, kp_pd, transpose_x=True)
        #print(kptv_pd)
        y_pd = paddle.matmul(qp_pd, kptv_pd, transpose_y=True)
        y_pd = y_pd / (D_pd.expand([D_pd.shape[0], D_pd.shape[1], emb]) + 1e-3)
        #print('y_pd: ', y_pd)

        # check if paddle out equals to pytorch out 
        D_pth_np = D_pth.cpu().numpy()
        D_pd_np = D_pd.cpu().numpy()
        self.assertTrue(np.allclose(D_pth_np, D_pd_np, rtol=1e-2))
        #print('D same')

        kptv_pth_np = kptv_pth.cpu().numpy()
        kptv_pd_np = kptv_pd.cpu().numpy()
        self.assertTrue(np.allclose(kptv_pth_np, kptv_pd_np, rtol=1e-2))
        #print('kptv same')

        y_pth_np = y_pth.cpu().numpy()
        y_pd_np = y_pd.cpu().numpy()
        self.assertTrue(np.allclose(y_pth_np, y_pd_np, rtol=1e-2))
        #print('y same')

    #@unittest.skip('skip for debug')
    def test_token_performer(self):
        tp = TokenPerformer(96, 96)


