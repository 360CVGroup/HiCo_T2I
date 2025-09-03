import os
import sys
import json
import pdb
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image

#from torchmetrics.functional.multimodal import clip_score
#from functools import partial
#clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("clip-vit-base-patch16",local_files_only=True,cache_dir="/root/.cache/huggingface/hub/")
processor = CLIPProcessor.from_pretrained("clip-vit-base-patch16", local_files_only=True,cache_dir="/root/.cache/huggingface/hub/")

gid = int(sys.argv[1])
#pid = int(sys.argv[2])
#tols = int(sys.argv[3])
ext = sys.argv[2]

file_json = "./detect_GDINO_result_coco3k_%s_All.json" % ext

cmd = "touch ./FT_mIoU_%s_begin.time" % ext
os.system(cmd)
out_file_base = "./FT_mIoU_%s_base.json" % ext
out_file_full = "./FT_mIoU_%s_full.json" % ext

model.to("cuda:%s" % gid)

root_dir = "/home/jovyan/boomcheng-data/aigc/LayoutProj/hico7k-val-512S/"

def cal_clip_score(path_image, crop_bbox, caption):
    image = Image.open(path_image)
    crop_image = image.crop(crop_bbox)

    #images_int = (np.asarray(crop_image) * 255).astype("uint8")
    #images_int = np.asarray(crop_image).astype("uint8")
    #clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    #return round(float(clip_score), 4)

    inputs = processor(text=caption, images=crop_image, return_tensors="pt", padding=True).to("cuda:%s" % gid)

    outputs = model(**inputs)

    #cos_sim = F.cosine_similarity(outputs.text_embeds, outputs.image_embeds)
    cos_sim = outputs.logits_per_image.item() / 100

    return cos_sim

def get_maximum_bbox(gt_bbox, list_bbox):
    gt_bbox = np.array(gt_bbox)
    list_bbox = np.array(list_bbox)

    ixmin = np.maximum(list_bbox[:, 0], gt_bbox[0])
    iymin = np.maximum(list_bbox[:, 1], gt_bbox[1])
    ixmax = np.minimum(list_bbox[:, 2], gt_bbox[2])
    iymax = np.minimum(list_bbox[:, 3], gt_bbox[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
            (list_bbox[:, 2] - list_bbox[:, 0] + 1.) *
            (list_bbox[:, 3] - list_bbox[:, 1] + 1.) - inters)
    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)

    return int(jmax), float(ovmax)


with open(file_json, encoding='utf-8') as f:
    json_data = json.load(f)


cnt = 0
out_data = {}
out_data_full = {}
for imgid, values in json_data[0].items():
    cnt += 1
    if cnt % 100 == 0:
        print ("proc : %s" % cnt)
    sub_mIoU = []
    sub_mIoU_full = []
    for d_info in values:
        #
        gt_caption, gt_bbox, list_pd_caption, list_pd_conf, list_pd_bbox = d_info

        d_caption = ""
        d_iou =None
        d_idx = None
        d_bbox = []
        d_conf = None
        clip_score = -1

        if len(list_pd_caption) == 0:
            # not generation
            pass
        else:
            if len(list_pd_bbox) > 1:
                idx, iou = get_maximum_bbox(gt_bbox, list_pd_bbox)
            else:
                idx, iou = get_maximum_bbox(gt_bbox, list_pd_bbox)
            d_caption, d_iou, d_idx, d_conf, d_bbox = list_pd_caption[idx], iou, idx, list_pd_conf[idx], list_pd_bbox[idx]

            # cal clip-score
            #path_image = "%s/%s.png" % (root_dir, imgid)
            path_image = "%s/%012d.png" % (root_dir, int(imgid))
            clip_score = cal_clip_score(path_image, d_bbox, gt_caption)

        sub_mIoU_full.append([gt_caption, gt_bbox, [d_caption, d_iou, clip_score, d_idx, d_conf, d_bbox], [list_pd_caption, list_pd_conf, list_pd_bbox]])

        sub_mIoU.append([gt_caption, gt_bbox, [d_caption, d_iou, clip_score, d_idx, d_conf, d_bbox]])

    out_data.setdefault(imgid, sub_mIoU)
    out_data_full.setdefault(imgid, sub_mIoU_full)

with open(out_file_base, "w", encoding="utf-8") as f:
    f.write(json.dumps([out_data], ensure_ascii=False, indent=4, separators=(',', ':')))

with open(out_file_full, "w", encoding="utf-8") as f:
    f.write(json.dumps([out_data_full], ensure_ascii=False, indent=4, separators=(',', ':')))
