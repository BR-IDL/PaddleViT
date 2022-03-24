import numpy as np
import paddle.inference as paddle_infer

config = paddle_infer.Config('./beit_base.pdmodel',
                             './beit_base.pdiparams')
predictor = paddle_infer.create_predictor(config)

input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

fake_input = np.random.randn(4, 3, 224, 224).astype('float32')
input_handle.reshape([4, 3, 224, 224])
input_handle.copy_from_cpu(fake_input)

predictor.run()


output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
print(output_data)
print('output shape = ', output_data.shape)
