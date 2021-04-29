import torch
import torch.nn.functional as F
import time
import os
import numpy as np
from PIL import Image
import cv2
# import cupy as cp
# import cupyx.scipy.ndimage as ndimage 


def image_tensor_to_array(image_tensor, mean, std):
    img = image_tensor.cpu().numpy()
    # chw->hwc
    img = np.transpose(img, (1, 2, 0))
    img *= np.array(std)
    img += np.array(mean)
    img *= 255
    img = img.astype(np.uint8)
    img = img[:, :, [2, 1, 0]]
    return img  

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[1], image.shape[0]
    # image = cv2.UMat(image)
    # iw, ih = image.shape[1], image.shape[0]
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    # image = cp.array(image)

    # trans_mat = cp.eye(4)
    # trans_mat[0][0] = trans_mat[1][1] = scale

    # smaller_shape = (nh, nw, 3)
    # smaller = cp.zeros(smaller_shape) # preallocate memory for resized image
    # t = time.time()
    # ndimage.interpolation.affine_transform(image, trans_mat, output_shape=smaller_shape,
    #                                     output=smaller, mode='nearest')
    # print('cupy resize time:', time.time() - t)

    # image = cp.asnumpy(smaller)
    # image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC).get()
    # t = time.time()
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    # print('cv2 resize time:', time.time() - t)

    new_image = np.zeros((h, w, 3), np.uint8) + 128
    new_image[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw, :] = image
    shift = [(w-nw)//2, (h-nh)//2]
    return new_image, scale, shift



# def letterbox_image(image, size):
#     '''resize image with unchanged aspect ratio using padding'''
#     iw, ih = image.size
#     w, h = size
#     scale = min(w/iw, h/ih)
#     nw = int(iw*scale)
#     nh = int(ih*scale)

#     image = image.resize((nw,nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128,128,128))
#     new_image.paste(image, ((w-nw)//2, (h-nh)//2))
#     shift = [(w-nw)//2, (h-nh)//2]
#     return new_image, scale, shift


def set_params(params, network, weight_decay, lr):
    params_dict = dict(network.named_parameters())
    for key, value in params_dict.items():
        if key[-4:] == 'bias':
            params += [{'params': value, 'weight_decay': 0.0, 'lr': lr * 2}]
        else:
            params += [{'params': value, 'weight_decay': weight_decay, 'lr': lr}]
    return params


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load success !')


def save_model(model, save_path, name, iter_cnt, use_multi_gpu):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    # multi gpu training
    if use_multi_gpu:
        torch.save(model.module.state_dict(), save_name)
    else:
        torch.save(model.state_dict(), save_name)

    return save_name


def save_checkpoint(model, optimizer, save_path, epoch, use_multi_gpu):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_path = os.path.join(save_path, str(epoch))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_name = os.path.join(checkpoint_path, 'model.pth')
    optimizer_name = os.path.join(checkpoint_path, 'optimizer.pth')

    if model is not None:
        if use_multi_gpu:
            torch.save(model.module.state_dict(), model_name)
        else:
            torch.save(model.state_dict(), model_name)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_name)


def draw_net_graph(model):
    import hiddenlayer as hl
    g = hl.build_graph(model, torch.zeros([1, 1, 128, 128]))
    g.save(os.path.join('./', "resnetface20.pdf"))


def moving_average(net1, net2, alpha=1.0):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    return img


def get_output_size(model, shape):
    batch_size = 1  # Not important.
#        input = Variable(torch.rand(batch_size, *shape), requires_grad=False)
    input_data = torch.rand(batch_size, *shape, requires_grad=False)
    output_feat = self.features(input_data)
    flattened_size = self.num_flat_features(output_feat)
    return flattened_size


def remove_small_boxes(bboxes, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        bboxes (bbox tensors)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    # xywh_boxes = boxlist.convert("xywh").bbox
    # _, _, ws, hs = xywh_boxes.unbind(dim=1)

    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero(as_tuple=False).squeeze(1)
    # print('remove_small_boxes', keep.shape)
    return keep
