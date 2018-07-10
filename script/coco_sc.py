from personlab.datamaker.coco import COCOMaker
import skimage.io as io
import pickle


inst_json = '/home/ubuntu/coco/annotations/instances_train2017.json'
person_json = '/home/ubuntu/coco/annotations/person_keypoints_train2017.json'
coco_maker = COCOMaker(inst_json, person_json)


base = '/home/ubuntu/coco/train2017/'
cnt = 0
for img_info in coco_maker.img_infos:
    img = io.imread(base + img_info['file_name'])
    if len(img.shape) != 3:
        continue
    num_frame, frame_res, seg_res, kp_list, com = coco_maker.read_coco(base, img, img_info['id'])
    
    dset = {
        'f': frame_res,
        'seg': seg_res,
        'kp': kp_list
    }
    filename = 'coco_%d.mydat' % (cnt)
    cnt += 1
    with open('tr/' + filename, 'wb') as f:
        pickle.dump(dset, f)
