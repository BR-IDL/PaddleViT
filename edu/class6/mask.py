import paddle
from PIL import Image
paddle.set_device('cpu')

def windows_partition(x, window_size):
    """ partite windows into window_size x window_size
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """

    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([-1, window_size, window_size, C]) #(num_windows*B, window_size, window_

    return x


def generate_mask(window_size=4, shift_size=2, input_resolution=(8, 8)):
    H, W = input_resolution
    img_mask = paddle.zeros([1, H, W, 1])
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
   
    windows_mask = windows_partition(img_mask, window_size=window_size)
    windows_mask = windows_mask.reshape([-1, window_size*window_size])
    #[num_windows, ws*ws]
    attn_mask = windows_mask.unsqueeze(1) - windows_mask.unsqueeze(2) 
    #[n, 1, ws*ws] - [n, ws*ws, 1] = [n, ws*ws, ws*ws]
    attn_mask = paddle.where(attn_mask!=0,
                             paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
    return attn_mask


def main():
    mask = generate_mask()
    print(mask.shape)
    mask = mask.cpu().numpy().astype('uint8')
    for i in range(4):
        for j in range(16):
            for k in range(16):
                print(mask[i,j,k], end='\t')
            print()
        im = Image.fromarray(mask[i, :, :])
        im.save(f'{i}.png')
        print()
        print()
    print()


if __name__ == "__main__":
    main()
