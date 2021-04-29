import torch
import cv2
from torchvision import transforms as T
from torchvision import ops
import yaml
import numpy as np
import time

from model.atss import ATSSModel, prepare_targets
from utils import letterbox_image, remove_small_boxes, visualize
from bbox.bbox import box_decode


def detect_images(model, model_inputs, scale, shift):

    t1 = time.time()
    with torch.no_grad():
        model_outputs = model(model_inputs)
    print('model infer time:', time.time() - t1)

    t1 = time.time()
    anchors = torch.cat(model.anchors, dim=0).cpu()

    det_results = []
    for i in range(model_inputs.shape[0]):
        model_output = [each[i].cpu() for each in model_outputs]
        det_per_image = post_process(model_output, anchors, scale[i], shift[i], model.num_classes)
        det_results.append(det_per_image)
    print('post_process time:', time.time() - t1)
    return det_results


def prepare_infer_input(img_path, input_size, transform):
    t = time.time()
    img = cv2.imread(img_path)
    # print('read file time:', time.time() - t)
    # keep ratio resize to input size
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = time.time()
    new_img, scale, shift = letterbox_image(new_img, input_size)
    # print('letterbox_image time:', time.time() - t)

    # t = time.time()
    # model_input = torch.Tensor(new_img).cuda()
    # print('image to cuda time:', time.time() - t)

    # mean = torch.Tensor(mean).cuda()
    # std = torch.Tensor(std).cuda()

    t = time.time()
    # model_input = (model_input / 255.0 - mean) / std
    # model_input = model_input.permute([2, 0, 1]).unsqueeze(0)
    model_input = transform(new_img).unsqueeze(0)
    # print('transform time:', time.time() - t)

    return img, model_input, scale, shift

def detect_single_image(model, model_input, scale, shift):

    t1 = time.time()
    with torch.no_grad():
        model_outputs = model(model_input)
    print('model infer time:', time.time() - t1)

    t1 = time.time()
    anchors = torch.cat(model.anchors, dim=0).cpu()
    model_outputs = [each.cpu() for each in model_outputs]
    det_result = post_process(model_outputs, anchors, scale, shift, model.num_classes)
    print('post_process time:', time.time() - t1)

    return det_result


def post_process(model_output, anchors, scale, shift, num_classes, pre_nms_thresh=0.05, pre_nms_top_n=1000, post_top_n=20, nms_iou_thres=0.5):
    # print(scale, shift)
    bbox_cls_logits, bbox_reg, centerness = model_output
    # print(bbox_cls_logits.shape, bbox_reg.shape, centerness.shape)

    # t = time.time()

    # bbox_cls_logits = bbox_cls_logits.cpu()
    # bbox_reg = bbox_reg.cpu()
    # centerness = centerness.cpu()
    # anchors = anchors.cpu()
    # print('t1 time:', time.time() - t)

    # t = time.time()

    bbox_cls_prob = bbox_cls_logits.reshape(-1, num_classes).sigmoid()

    # print('bbox_cls_prob', bbox_cls_prob)
    candidate_inds = bbox_cls_prob > pre_nms_thresh
    pre_nms_num = candidate_inds.view(-1).sum()
    pre_nms_num = pre_nms_num.clamp(max=pre_nms_top_n)
    # print("candidate_inds", candidate_inds)

    centerness = centerness.reshape(-1, 1).sigmoid()
    bbox_scores = (bbox_cls_prob * centerness)[candidate_inds]
    # bbox_scores = bbox_cls_prob[candidate_inds]

    bbox_scores, top_k_indices = bbox_scores.topk(pre_nms_num, sorted=False)
    candidate_nonzeros = candidate_inds.nonzero(as_tuple=False)[top_k_indices, :]

    bbox_loc = candidate_nonzeros[:, 0]
    bbox_labels = candidate_nonzeros[:, 1]
    # print('t1 time:', time.time() - t)

    # bbox_reg = bbox_reg.reshape(-1, 4)[box_loc, :]
    # anchors = anchors[box_loc, :]
    # t = time.time()
    detections = box_decode(bbox_reg.reshape(-1, 4)[bbox_loc, :], anchors[bbox_loc, :])
    # print('t1 time:', time.time() - t)

    # remove small boxes
    keep = remove_small_boxes(detections, min_size=0)
    detections = detections[keep]
    bbox_labels = bbox_labels[keep]
    # bbox_scores = bbox_scores[keep]
    bbox_scores = torch.sqrt(bbox_scores[keep])
    # print('t1 time:', time.time() - t)

    # nms
    # t = time.time()
    keep = ops.nms(detections, bbox_scores, nms_iou_thres)
    # print('nms time:', time.time() - t)
    # print("after nms keep", keep.shape)

    # t = time.time()

    # detections = detections.cpu()
    # bbox_labels = bbox_labels.cpu()
    # bbox_scores = bbox_scores.cpu()


    detections = detections[keep]
    bbox_labels = bbox_labels[keep]
    bbox_scores = bbox_scores[keep]

    # top 20
    post_top_n = min(keep.shape[0], post_top_n)
    detections = detections[:post_top_n]
    bbox_labels = bbox_labels[:post_top_n]
    bbox_scores = bbox_scores[:post_top_n]
    # print("bbox_scores", bbox_scores)

    # scale & shift back
    detections[:, [0, 2]] = detections[:, [0, 2]] - shift[0]
    detections[:, [1, 3]] = detections[:, [1, 3]] - shift[1]
    detections = detections / scale
    # print('t2 time:', time.time() - t)

    det_result = []
    for i in range(num_classes):
        is_cls_i = (bbox_labels == i)

        try:
            bboxes = detections[is_cls_i]
            scores = bbox_scores[is_cls_i]
        except Exception:
            print(Exception)
            print('detections, bbox_scores', detections, bbox_scores)
            print('is_cls_i', is_cls_i)

        det_bboxes = torch.cat([bboxes, scores[:,None]], dim=1)
        det_result.append(det_bboxes.numpy())

    return det_result   


def draw_bbox_save_image(image, det_results, save_file, category_id_to_name):
    bboxes = []
    bbox_labels = []
    for i in range(len(det_results)):
        bboxes.append(det_results[i])
        bbox_labels.append(np.full(len(det_results[i]), i))
    bboxes = np.concatenate(bboxes)[:, :4]
    bbox_labels = np.concatenate(bbox_labels)

    bbox_img = visualize(image, bboxes=bboxes, category_ids=bbox_labels, category_id_to_name=category_id_to_name)
    cv2.imwrite(save_file, bbox_img)


def main():
    print('cuda avilable:', torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gpu_num = torch.cuda.device_count()
    # cv2.setNumThreads(16)

    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    cls_names = cfg['class_names']
    category_id_to_name = {k: v for k, v in enumerate(cls_names)}

    model_path = 'checkpoints/1020/160/model.pth'
    num_classes = len(cls_names)

    input_size = cfg['image_size']
    model = ATSSModel(input_size=[3, *input_size], num_classes=len(cls_names))
    status = model.load_state_dict(torch.load(model_path, map_location=device))
    print(status)
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor(),
                           T.Normalize(cfg['imagenet_default_mean'], cfg['imagenet_default_std'])])

    img_path = 'data/images/gmnnc-bs/gmnnc-bs-2020092701/1/ch04_20200924140031_105334_779211.jpg'
    t1 = time.time()
    # orig_img, model_input, scale, shift = prepare_infer_input(img_path, input_size, transform, cfg['imagenet_default_mean'], cfg['imagenet_default_std'])
    orig_img, model_input, scale, shift = prepare_infer_input(img_path, input_size, transform)

    print('prepare_infer_input time:', time.time() - t1)

    det_results = infer_single_image(model, model_input.to(device), scale, shift)

    draw_bbox_save_image(orig_img, det_results, 'result_1.jpg', category_id_to_name)

if __name__ == '__main__':
    main()