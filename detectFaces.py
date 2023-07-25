import cv2
import os
import pandas as pd
from retinaface.RetinaFace import detect_faces
import glob
import re


class FaceDetection:
    def __init__(self):
        pass

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def extractFaces(self, list_of_images):
        saveFacesHere = "Faces/"
        savePaddedFaces = "PaddedFaces/"
        if not os.path.exists(saveFacesHere):
            os.makedirs(saveFacesHere)

        if not os.path.exists(savePaddedFaces):
            os.makedirs(savePaddedFaces)

        df = pd.DataFrame(
            columns=["FrameFileName", "FacesPath", "PaddedFacesPath", "FA1", "FA2", "FA3", "FA4"])

        def get_faces(image_path):
            return detect_faces(image_path)

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
                resp = get_faces(image_path)
                print(f"**{image_path} : {resp}")
                img = cv2.imread(image_path)
                image_wid = img.shape[1]
                image_hgt = img.shape[0]
                i = 0
                for key, value in resp.items():
                    tmp = []
                    aa = value['facial_area']
                    FA = [aa[1], aa[2], aa[3], aa[0]]
                    distnce_between_rightleft_eye = abs(
                        value['landmarks']['right_eye'][0] - value['landmarks']['left_eye'][0])
                    if distnce_between_rightleft_eye < 25:
                        pass
                    else:
                        x1, y1, x2, y2 = aa[0], aa[1], aa[2], aa[3]
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
        try:
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
            extract_faces_df = self.extractFaces(list_of_images)
            extract_faces_df.to_csv("facesInfo.csv")

        except Exception as e:
            print(e)
            raise


obj = FaceDetection()
obj.faceDetection("images")
