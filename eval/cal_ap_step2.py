import os
import sys
import json
import pdb
import numpy as np


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 为求得曲线面积，对 Precision-Recall plot进行“平滑”处理
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 找到 recall值变化的位置
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # recall值乘以相对应的 precision 值相加得到面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

ext = sys.argv[1]

IOU_THR = float(sys.argv[2])
CLIP_SCORE_THR = float(sys.argv[3])


file_json = "./FT_mIoU_%s_full.json" % ext

with open(file_json, encoding='utf-8') as f:
    json_data = json.load(f)

list_clip_score = []
list_iou = []
objs_gt = 0
list_subs_info = []
for imgid, values in json_data[0].items():
    for sub_info in values:
        # base
        #gt_caption, gt_bbox, pred_info = sub_info
        #pred_caption, pred_iou, pred_idx, pred_conf, pred_bbox = pred_info
        # full
        gt_caption, gt_bbox, pred_info, list_pred_info = sub_info
        pred_caption, pred_iou, pred_clip_score, pred_idx, pred_conf, pred_bbox = pred_info

        objs_gt += 1

        if len(pred_bbox) == 0:
            pass
        else:
            #list_subs_info.append([imgid, gt_caption, gt_bbox, pred_caption, pred_iou, pred_idx, pred_conf, pred_bbox])
            list_subs_info.append([imgid, gt_caption, gt_bbox, pred_caption, pred_iou, pred_clip_score, pred_idx, pred_conf, pred_bbox])

            list_clip_score.append(pred_clip_score)
            list_iou.append(pred_iou)

print ("images: %s,   objects  gt: %s, pred:%s" % (len(json_data[0]), objs_gt, len(list_subs_info)))

print ("avg clip-score : %.4f, nums:%s" % (sum(list_clip_score) / len(list_clip_score), len(list_clip_score) ))
print ("avg iou-score : %.4f, nums:%s" % (sum(list_iou) / len(list_iou), len(list_iou) ))
st_list_subs_info = sorted(list_subs_info, key=lambda x:x[6], reverse=True)


#IOU_THR = 0.5
#CLIP_SCORE_THR = 0.2

list_AR = []
list_AP = []
list_IOU_THR = [i/100 for i in range(50,100,5)]
#pdb.set_trace()
for IOU_THR in list_IOU_THR:

    TP = 0
    FP = 0

    precision = []
    recall = []

    for dot in st_list_subs_info:
        #imgid, gt_caption, gt_bbox, pred_caption, pred_iou, pred_idx, pred_conf, pred_bbox = dot
        imgid, gt_caption, gt_bbox, pred_caption, pred_iou, pred_clip_score, pred_idx, pred_conf, pred_bbox = dot
        if pred_iou > IOU_THR and pred_clip_score > CLIP_SCORE_THR:
            # TP
            TP += 1
        else:
            FP += 1
        
        d_prec = TP / (TP + FP)
        d_recall = TP / (objs_gt)

        precision.append(d_prec)
        recall.append(d_recall)

    ar = TP / objs_gt
    ap = voc_ap(recall, precision)

    print (IOU_THR, ap, ar)
    list_AP.append(ap)
    list_AR.append(ar)

print ("AP: %.4f" % (sum(list_AP) / len(list_AP)))
print ("AR: %.4f" % (sum(list_AR) / len(list_AR)))


