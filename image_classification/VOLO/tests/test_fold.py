import unittest
import numpy as np
import paddle
import paddle.nn as nn
from fold import fold


class FoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
    
    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_fold_1(self):
        """test padding=2, stride=2"""
        arr = [
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
        ]
        arr = np.array(arr)
        tmp = paddle.to_tensor(arr, dtype='float32')
        tmp = tmp.reshape([1, 1, 8, 8])
    
        unfold = paddle.nn.Unfold(3, 2, 2)
        out = unfold(tmp)
        out = fold(out, output_size=(8, 8), kernel_size=3, padding=2, stride=2)
        ans = [[[[4. , 2. , 4. , 2. , 8. , 4. , 8. , 4. ],
                 [2. , 1. , 2. , 1. , 4. , 2. , 4. , 2. ],
                 [4. , 2. , 4. , 2. , 8. , 4. , 8. , 4. ],
                 [2. , 1. , 2. , 1. , 4. , 2. , 4. , 2. ],
                 [12., 6. , 12., 6. , 16., 8. , 16., 8. ],
                 [6. , 3. , 6. , 3. , 8. , 4. , 8. , 4. ],
                 [12., 6. , 12., 6. , 16., 8. , 16., 8. ],
                 [6. , 3. , 6. , 3. , 8. , 4. , 8. , 4. ]]]]
        self.assertTrue(np.allclose(np.array(ans), out.numpy()))

    def test_fold_2(self):
        """test padding=1, stride=1"""
        arr = [
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
           [3, 3, 3, 3, 4, 4, 4, 4],
        ]
        arr = np.array(arr)
        tmp = paddle.to_tensor(arr, dtype='float32')
        tmp = tmp.reshape([1, 1, 8, 8])
    
        unfold = paddle.nn.Unfold(3, 1, 1)
        out = unfold(tmp)
        out = fold(out, output_size=(8, 8), kernel_size=3, padding=1, stride=1)
        ans = [[[[4. , 6. , 6. , 6. , 12., 12., 12., 8. ],
                 [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
                 [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
                 [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
                 [18., 27., 27., 27., 36., 36., 36., 24.],
                 [18., 27., 27., 27., 36., 36., 36., 24.],
                 [18., 27., 27., 27., 36., 36., 36., 24.],
                 [12., 18., 18., 18., 24., 24., 24., 16.]]]]

        self.assertTrue(np.allclose(np.array(ans), out.numpy()))
