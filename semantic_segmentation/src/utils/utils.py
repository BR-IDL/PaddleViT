

import contextlib
import filelock
import math
import os
import paddle.nn.functional as F

import tempfile
from urllib.parse import urlparse, unquote

import paddle

from src.utils import logger, seg_env
from src.utils.download import download_file_and_uncompress


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    """
    Load the weights of the whole model
    
    Arges:
       model: model based paddle
       pretrained: the path of weight file of model 
    """

    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Not all pretrained params of {} are loaded, ' \
                       'training from scratch or a pretrained backbone.'.format(model.__class__.__name__))


def load_pretrained_model(model, pretrained_model, pos_embed_interp=True):
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            # 4 debug
            #print("pretrain_model.param: ", para_state_dict.keys())
            #print("pretrained para_state_dict.len: ", len(para_state_dict.keys()))
            #print("current model weight.len: ",len(keys))
            match_list=[]
            not_match_list=[]
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                    not_match_list.append(k)
                elif list(para_state_dict[k].shape) != list(model_state_dict[k].shape):
                    if pos_embed_interp==True:
                        n, pretrain_num_patches, c = para_state_dict[k].shape  # pretrain_num_patches=hw+1
                        n, cur_model_num_patches, c = model_state_dict[k].shape
                        h = w = int(math.sqrt(pretrain_num_patches))
                        pos_embed_weight = para_state_dict[k][:, (-h * w):]  # (n,hw,c)
                        pos_embed_weight = pos_embed_weight.transpose([0,2,1])  # (n,c,hw)
                        pos_embed_weight = pos_embed_weight.reshape([n, c, h, w])  # type: numpy
                        pos_embed_weight = paddle.to_tensor(pos_embed_weight)
                        cur_h=int(math.sqrt(cur_model_num_patches))
                        cur_pos_embed_weight = F.interpolate(pos_embed_weight, size=(cur_h, cur_h), mode='bilinear', align_corners=False)
                        cur_pos_embed_weight = cur_pos_embed_weight.reshape([n, c, -1]).transpose([0,2,1])
                        cls_token_weight = para_state_dict[k][:, 0]
                        cls_token_weight = paddle.to_tensor(cls_token_weight).unsqueeze(1)
                        model_state_dict[k] = paddle.concat((cls_token_weight, cur_pos_embed_weight), axis=1).numpy()
                        num_params_loaded += 1
                        match_list.append(k)
                    else:
                        logger.warning("[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                            .format(k, para_state_dict[k].shape,
                            model_state_dict[k].shape))
                    
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
                    match_list.append(k)

            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), model.__class__.__name__))
            logger.info(" {} parameters is matched, and {} parameters is Not matched".format(
                len(match_list), len(not_match_list)))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        logger.info('No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            iter = resume_model.split('_')[-1]
            iter = int(iter)
            return iter
        else:
            raise ValueError('Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')
