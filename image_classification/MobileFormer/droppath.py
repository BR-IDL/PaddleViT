# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MobileFormer Arch -- DroPPath Implement
"""
import paddle
from paddle import nn

import numpy as np

class DropPath(nn.Layer):
    """Multi-branch dropout layer -- Along the axis of Batch
        Params Info:
            p: droppath rate
    """
    def __init__(self,
                 p=0.):
        super(DropPath, self).__init__(
                 name_scope="DropPath")
        self.p = p
    
    def forward(self, inputs):
        if self.p > 0. and self.training:
            keep_p = np.asarray([1 - self.p], dtype='float32')
            keep_p = paddle.to_tensor(keep_p)
            # B, 1, 1....
            shape = [inputs.shape[0]] + [1] * (inputs.ndim-1)
            random_dr = paddle.rand(shape=shape, dtype='float32')
            random_sample = paddle.add(keep_p, random_dr).floor() # floor to int--B
            output = paddle.divide(inputs, keep_p) * random_sample
            return output

        return inputs