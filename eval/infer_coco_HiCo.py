#from groundingdino.util.inference import load_model, load_image, predict, ann
from groundingdino.util.inference import load_model, load_image, predict
import cv2
import os
import sys
import numpy as np
import json,jsonlines
import supervision as sv
import torch
from torchvision.ops import box_convert
import json
import pdb
import PIL 
from PIL import Image, ImageFont, ImageDraw
import time
import datetime

def _load_image(image_path):
    with open(image_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            image = image.convert('RGB')
    return image

def draw_image_xyxy(image, obj_bbox, obj_class, img_save):
    dw_img = PIL.Image.fromarray(np.uint8(image))
    #dw_img = PIL.Image.fromarray(np.uint8(image * 255))

    draw = PIL.ImageDraw.Draw(dw_img)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
    for iix in range(len(obj_bbox)):
        rec = obj_bbox[iix]
        d_rec = [int(xx) for xx in rec]
        draw.rectangle(d_rec, outline = color, width = 3)

        text = obj_class[iix]
        font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=14)
        draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
    dw_img.save(img_save)

def draw_image_xywh(image, obj_bbox, obj_class, img_save):
    dw_img = PIL.Image.fromarray(np.uint8(image))
    #dw_img = PIL.Image.fromarray(np.uint8(image * 255))
    draw = PIL.ImageDraw.Draw(dw_img)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
    for iix in range(len(obj_bbox)):
        rec = obj_bbox[iix]
        d_rec = [int(xx) for xx in rec]
        d_rec[2] += d_rec[0]
        d_rec[3] += d_rec[1]
        draw.rectangle(d_rec, outline = color, width = 3)

        text = obj_class[iix]
        font = ImageFont.truetype("./font/msyh.ttf", size=14)
        draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
    dw_img.save(img_save)



file_json = "/home/jovyan/boomcheng-data/aigc/LLM/G-DINO/grit-20m-hico7k-val-7k-512s.json"
with open(file_json, encoding='utf-8') as f:
    json_data = json.load(f)

dict_img_info = {}
for k,v in json_data.items():   
    caption, list_class, list_bbox = v
    img_id = int(k.split(".")[0])
    obj_class = list_class
    obj_bbox = [list(map(lambda y:y*512, x)) for x in list_bbox]

    obj_bbox = np.array(obj_bbox)
    dict_img_info.setdefault(img_id, [obj_class, obj_bbox])

print (len(dict_img_info))

gid = int(sys.argv[1])
pid = int(sys.argv[2])
tols = int(sys.argv[3])

device="cuda:%s" % gid
model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth")

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# GT 
#base_ckpt_info = "image_grit_20m_val_512P_GT"
#dir_in = "/home/jovyan/boomcheng-data/aigc/LayoutProj/GRIT/image_grit-20m-val-512S/"

#base_ckpt_info = "image_coco3k_512P_InstDiff"
#dir_in = "/home/jovyan/boomcheng-data/aigc/LayoutProj/diffusers/examples/controlnet/result_coco3k/image_coco_SD15_checkpoint-500000_UniPCM_cfg7.5_512P/"
#out_file = "detect_GDINO_result_coco3k_InstDiff_%03d.json" % pid

base_ckpt_info = "image_coco3k_Real_512P_HiCo"
dir_in = "/home/jovyan/boomcheng-data/aigc/LayoutProj/diffusers/result_hico7k/image_coco_RealV51_checkpoint-500000_UniPCM_cfg7.5_512P/"
out_file = "detect_GDINO_result_coco3k_Real_HiCo_%03d.json" % pid


dir_save = "./%s_GDINO" % base_ckpt_info
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

list_imgs = os.listdir(dir_in)

dict_out_info = {}
cnt = 0
for img_name in list_imgs:
    cnt += 1
    if cnt % tols != pid:
        continue
    if cnt % 1000 == 0:
        curdate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print ("%s proc-%s %s" % (curdate, pid, cnt))
    #img_name = line.strip()
    img_id = int(img_name.split(".")[0])
    if img_id not in dict_img_info:
        print ("Not imgid exists. %s" % img_id)
        continue

    #pdb.set_trace()
    path_img = os.path.join(dir_in, img_name)
    gt_obj_class, gt_obj_bbox = dict_img_info[img_id]

    #img_save = "./%s/%s-gt.jpg" % (dir_save, img_id)
    #draw_image_xyxy(image_source, gt_obj_bbox, gt_obj_class, img_save)

    out_det = []
    for ii in range(len(gt_obj_class)):
        #pdb.set_trace()
        dd_obj_class = gt_obj_class[ii]
        dd_obj_bbox = gt_obj_bbox[ii].astype(np.int64).tolist()
        try:
            image_source, image = load_image(path_img)

            TEXT_PROMPT = dd_obj_class
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                #device=device,
            )

            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            obj_bbox = xyxy.astype(np.int64).tolist()
            #obj_class = [phrases[0]] * len(obj_bbox)
            if len(phrases) != len(obj_bbox):
                #obj_class = phrases * len(obj_bbox)
                obj_class = [phrases[0]] * len(obj_bbox)
            else:
                obj_class = phrases

            img_save = "%s/%s_%03d.jpg" % (dir_save, img_id, pid)
            #draw_image_xyxy(image_source, obj_bbox, obj_class, img_save)
            logits_r = logits.numpy().tolist()

            out_det.append([dd_obj_class, dd_obj_bbox, obj_class, logits_r, obj_bbox])
        except Exception as err:
            print (img_name)
            print (err)
            continue

    dict_out_info.setdefault(img_id, out_det)
    #if cnt > 50:
    #    break


with open(out_file, "w", encoding="utf-8") as f:
    f.write(json.dumps([dict_out_info], ensure_ascii=False, indent=4, separators=(',', ':')))

