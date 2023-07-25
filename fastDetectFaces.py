import cv2
import os
import pandas as pd
import glob
import time
import numpy as np
from cv2 import dnn
from math import ceil
import re

class FaceDetection:
    def __init__(self):
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
        self.strides = [8.0, 16.0, 32.0, 64.0]

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)
        return priors

    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        print("priors nums:{}".format(len(priors)))
        return np.clip(priors, 0.0, 1.0)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def area_of(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
                                      iou_threshold=iou_threshold,
                                      top_k=top_k,
                                      )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def convert_locations_to_boxes(self, locations, priors, center_variance,
                                   size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    def center_form_to_corner_form(self, locations):
        return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                               locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

    def inference(self, net, img_path):
        # onnx version
        input_size = [int(v.strip()) for v in "320,240".split(",")]
        witdh = input_size[0]
        height = input_size[1]
        priors = self.define_img_size(input_size)
        # result_path = resultPath
        # imgs_path = input_images_folder
        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        # listdir = os.listdir(imgs_path)
        # for file_path in listdir:
        #     img_path = os.path.join(imgs_path, file_path)
        img_ori = img_path
        rect = cv2.resize(img_ori, (witdh, height))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        net.setInput(dnn.blobFromImage(rect, 1 / self.image_std, (witdh, height), 127))
        time_time = time.time()
        boxes, scores = net.forward(["boxes", "scores"])
        print("inference time: {} s".format(round(time.time() - time_time, 4)))
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = self.convert_locations_to_boxes(boxes, priors, self.center_variance, self.size_variance)
        boxes = self.center_form_to_corner_form(boxes)
        boxes, labels, probs = self.predict(img_ori.shape[1], img_ori.shape[0], scores, boxes, 0.3)
        return boxes
        # for i in range(boxes.shape[0]):
        #     box = boxes[i, :]
        #     cv2.rectangle(img_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(result_path, file_path), img_ori)

    def extractFaces(self, list_of_images, model):
        saveFacesHere = "Faces/"
        savePaddedFaces = "PaddedFaces/"
        if not os.path.exists(saveFacesHere):
            os.makedirs(saveFacesHere)

        if not os.path.exists(savePaddedFaces):
            os.makedirs(savePaddedFaces)

        df = pd.DataFrame(
            columns=["FrameFileName", "FacesPath", "PaddedFacesPath", "FA1", "FA2", "FA3", "FA4"])

        for image_path in list_of_images:
            try:
                ry = image_path.split("\\")
                ry = ry[len(ry) - 1]
                z = str(ry).split(".")[0]
            except:
                ry = image_path.split("/")
                ry = ry[len(ry) - 1]
                z = str(ry).split(".")[0]
            print(z)
            try:
                img = cv2.imread(image_path)
                # resp = model.predict_jsons(img)
                resp = self.inference(model, img)
                resp = resp.tolist()
                print(f"**{image_path} : {resp}")
                image_wid = img.shape[1]
                image_hgt = img.shape[0]
                i = 0
                for ii in resp:
                    tmp = []
                    # aa = ii['bbox']
                    FA = [int(ii[1]), int(ii[2]), int(ii[3]), int(ii[0])]
                    # distnce_between_rightleft_eye = abs(
                    #     ii['landmarks'][0][0] - ii['landmarks'][1][0])
                    # if distnce_between_rightleft_eye < 25:
                    #     pass
                    # else:
                    x1, y1, x2, y2 = int(ii[0]), int(ii[1]), int(ii[2]), int(ii[3])
                    x = x1
                    y = y1
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    crop_img = img[y:y + h, x:x + w]
                    wid = crop_img.shape[1]
                    hgt = crop_img.shape[0]
                    if (x + w + 50) <= image_wid:
                        croped_hight = y + h + 50
                    else:
                        croped_hight = y + h
                    if (y + h + 50) <= image_hgt:
                        croped_width = x + w + 50
                    else:
                        croped_width = x + w
                    crop_img_clip = img[y - 30:croped_hight, x - 30:croped_width]
                    if abs(wid - hgt) < 15:
                        pass
                    else:
                        cv2.imwrite(saveFacesHere + str(z) + "_" + str(i) + '.png', crop_img)
                        try:
                            cv2.imwrite(savePaddedFaces + str(z) + "_" + str(i) + '.png', crop_img_clip)
                        except:
                            cv2.imwrite(savePaddedFaces + str(z) + "_" + str(i) + '.png', crop_img)
                        tmp.append(image_path)
                        tmp.append(saveFacesHere + str(z) + "_" + str(i) + '.png')
                        tmp.append(savePaddedFaces + str(z) + "_" + str(i) + '.png')
                        tmp.append(FA[0])
                        tmp.append(FA[1])
                        tmp.append(FA[2])
                        tmp.append(FA[3])
                        i += 1
                    if len(tmp) != 0:
                        df_length1 = len(df)
                        df.loc[df_length1] = tmp
            except Exception as e:
                # res[image_path] = []
                print('Error in cropping face :', e)
                pass

        return df

    def faceDetection(self, images_path):
        # https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

        try:
            model = "fastModels/onnx/version-RFB-320_simplified.onnx"
            net = dnn.readNetFromONNX(model)
            list_of_images = []
            isFolder = True
            tm = images_path.split(".")
            if len(tm) > 1:
                isFolder = False
            if isFolder:
                for fi in glob.glob(images_path + "/*"):
                    list_of_images.append(fi)
            else:
                list_of_images.append(images_path)
            list_of_images.sort(key=self.natural_keys)
            extract_faces_df = self.extractFaces(list_of_images, net)
            extract_faces_df.to_csv("facesInfo.csv")
        except Exception as e:
            print(e)
            raise


obj = FaceDetection()
obj.faceDetection("images")