from pope_model_api import *


if __name__  == "__main__":
    ckpt, model_type = get_model_info("h")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    DEVICE = "cuda"
    sam.to(device=DEVICE)
    MASK_GEN = SamAutomaticMaskGenerator(sam)
    logger.info(f"load SAM model from {ckpt}")
    crop_tool = CropImage()
    dinov2_model = load_dinov2_model()
    dinov2_model.to("cuda:0")

    metrics = dict()
    metrics.update({'R_errs': [], 't_errs': [], 'inliers': [] , "identifiers":[] })


    # load data ROOR_DIR 
    ROOT_DIR = "data/ycbv/"
    res_table = []

    import json
    with open("data/pairs/YCB-VIDEO-test.json") as f:
        dir_list = json.load(f)

    for label_idx , test_dict in enumerate(dir_list):
        logger.info(f"YCBVIDEO: {label_idx}")
        metrics = dict()
        metrics.update({'R_errs': [], 't_errs': [], 'inliers': [] , "identifiers":[] })
        sample_data = dir_list[label_idx]["0"][0]
        label = sample_data.split("/")[0]
        name = label.split("-")[1]
        dir_name = os.path.dirname(sample_data)
        FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name) 
        recall_image,all_image = 0,0
        for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
            for pair_idx,pair_name in enumerate(tqdm(rotation_list[::2])):
                all_image = all_image + 1
                base_name = os.path.basename(pair_name)
                idx0_name = base_name.split("png-")[0]+"png"
                idx1_name = base_name.split("png-")[1]
                image0_name = os.path.join( FULL_ROOT_DIR, idx0_name )
                image1_name = os.path.join( FULL_ROOT_DIR.replace("color", "color_full"),  idx1_name )
                intrinsic_path = image0_name.replace("color", "intrin_ba").replace("png","txt")
    
                K0 = np.loadtxt(intrinsic_path, delimiter=' ')
                intrinsic_path = image1_name.replace("color_full", "intrin").replace("png","txt")
                K1 = np.loadtxt(intrinsic_path, delimiter=' ')
                image0 = cv2.imread(image0_name)
                ref_torch_image = set_torch_image(image0, center_crop=True)
                ref_fea = get_cls_token_torch(dinov2_model, ref_torch_image)
                image1 = cv2.imread(image1_name)
                image_h,image_w,_ = image1.shape
                t1 = time.time()
                masks = MASK_GEN.generate(image1)
                t2 = time.time()
                similarity_score, top_images  = np.array([0,0,0],np.float32) , [[],[],[]]
                t3 = time.time()
                compact_percent = 0.3
                for xxx, mask in enumerate(masks):
                    object_mask = np.expand_dims(mask["segmentation"], -1)
                    x0, y0, w, h = mask["bbox"]
                    x1, y1 = x0+w,y0+h
                    x0 -= int(w * compact_percent)
                    y0 -= int(h * compact_percent)
                    x1 += int(w * compact_percent)
                    y1 += int(h * compact_percent)
                    box = np.array([x0, y0, x1, y1])
                    resize_shape = np.array([y1 - y0, x1 - x0])
                    K_crop, K_crop_homo = get_K_crop_resize(box, K1, resize_shape)
                    image_crop, _ = get_image_crop_resize(image1, box, resize_shape)
                    # object_mask,_ = get_image_crop_resize(object_mask, box, resize_shape)
                    box_new = np.array([0, 0, x1 - x0, y1 - y0])
                    resize_shape = np.array([256, 256])
                    K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
                    image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
                    crop_tensor = set_torch_image(image_crop, center_crop=True)
                    with torch.no_grad():
                        fea = get_cls_token_torch(dinov2_model, crop_tensor)
                    score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)
                    if  (score.item() > similarity_score).any():
                        mask["crop_image"] = image_crop
                        mask["K"] = K_crop
                        mask["bbox"] = box
                        min_idx = np.argmin(similarity_score)
                        similarity_score[min_idx] = score.item()
                        top_images[min_idx] = mask.copy()

                img0 =  cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                img0 = torch.from_numpy(img0).float()[None] / 255.
                img0 = img0.unsqueeze(0).cuda()

                matching_score =  [ [0] for _ in range(len(top_images)) ]
                for top_idx in range(len(top_images)):
                    img1 =  cv2.cvtColor(top_images[top_idx]["crop_image"], cv2.COLOR_BGR2GRAY)
                    img1 = torch.from_numpy(img1).float()[None] / 255.
                    img1 = img1.unsqueeze(0).cuda()
                    batch = {'image0': img0, 'image1': img1}
                    with torch.no_grad():
                        matcher(batch)    
                        mkpts0 = batch['mkpts0_f'].cpu().numpy()
                        mkpts1 = batch['mkpts1_f'].cpu().numpy()
                        confidences = batch["mconf"].cpu().numpy()
                    conf_mask = np.where(confidences > 0.9)
                    matching_score[top_idx] = conf_mask[0].shape[0]
                    top_images[top_idx]["mkpts0"] = mkpts0
                    top_images[top_idx]["mkpts1"] = mkpts1
                    top_images[top_idx]["mconf"] = confidences
                #---------------------------------------------------
                # crop_image = cv2.resize(top_images[np.argmax(matching_score)]["crop_image"],(256,256))        
                # que_image = cv2.resize(image0,(256,256))
                # image = np.hstack((que_image, crop_image)) 
                # for top_idx in range(len(top_images)):
                #     crop_image = top_images[top_idx]["crop_image"]
                #     score = matching_score[top_idx]
                #     crop_image = cv2.resize(crop_image,(256,256))
                #     cv2.putText(crop_image,f'{score}',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                #     image = np.hstack((image, crop_image)) 
                # cv2.imwrite(f"segment_anything/crop_images/{idx}.jpg", image)
                #---------------------------------------------------
                t4 = time.time()
                # print(f"t4-t3: object detection:{1000*(t4-t3)} ms")
                pose0_name = image0_name.replace("color", "poses_ba").replace("png","txt")
                pose1_name = image1_name.replace("color_full", "poses_ba").replace("png","txt")
                pose0 = np.loadtxt(pose0_name)
                pose1 = np.loadtxt(pose1_name)
                pose0 = np.vstack((pose0, np.array([[0,0,0,1]])))
                pose1 = np.vstack((pose1, np.array([[0,0,0,1]])))
                relative_pose =  np.matmul(pose1, inv(pose0))
                t = relative_pose[:3,-1].reshape(1,3)

                max_match_idx = np.argmax(matching_score)
                pre_bbox  = top_images[max_match_idx]["bbox"]
                mkpts0 = top_images[max_match_idx]["mkpts0"]
                mkpts1 = top_images[max_match_idx]["mkpts1"]
                pre_K = top_images[max_match_idx]["K"]

                gt_bbox_name = image0_name.replace("color", "bbox_2d").replace("png","txt")
    
                gt_bbox = np.loadtxt(gt_bbox_name)
                is_recalled = recall_object(pre_bbox , gt_bbox)
                recall_image = recall_image + int(is_recalled>0.5)
                ret = estimate_pose(mkpts0, mkpts1 , K0 , pre_K , 0.5, 0.99)  
                if ret is  not None:
                    Rot, t, inliers = ret 
                    t_err, R_err = relative_pose_error(relative_pose, Rot, t, ignore_gt_t_thr=0.0)
                    metrics['R_errs'].append(R_err)
                    metrics['t_errs'].append(t_err)
                else:
                    metrics['R_errs'].append(90)
                    metrics['t_errs'].append(90)
                metrics["identifiers"].append( pair_name )

        import pprint
        from src.utils.metrics import (
            aggregate_metrics
        )
        from loguru import logger
        val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
        val_metrics_4tb["AP50"] = recall_image/all_image
        logger.info('\n' + pprint.pformat(val_metrics_4tb))
        res_table.append( [f"{name}"] +  list(val_metrics_4tb.values())  )

    from tabulate import tabulate
    headers = ["Category"] + list(val_metrics_4tb.keys())
    all_data = np.array(res_table)[:,1:].astype(np.float32)
    res_table.append( ["Avg"] + all_data.mean(0).tolist() )
    print(tabulate(res_table, \
        headers=headers, tablefmt='fancy_grid'))

