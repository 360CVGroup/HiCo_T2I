
gpu_id=0
pid=0
tols=1
# detect the obj and bbox
python infer_coco_HiCo.py ${gpu_id} ${pid} ${tols}

# cal IoU and AP/AR
python cal_ap_step1.py ${gpu_id} "HiCo7K"
python cal_ap_step2.py "HiCo7K"

