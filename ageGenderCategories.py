import clip
import torch
from PIL import Image, ImageDraw, ImageFont
from itertools import islice
import glob
import clipPrompts
from retinaface.RetinaFace import detect_faces
from detectFaces import FaceDetection
import re
import pandas as pd
import os
import cv2


class AgeGenderCategories:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def add_text_to_image(self, image, text, position=(20, 20), font_size=15, font_color=(199, 0, 57)):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(position, text, font=font, fill=font_color)

    # Function to resize images to a specified size
    def resize_image(self, image_path, target_size):
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        return image

    def create_collage(self, images, texts, canvas_size=(800, 600), image_size=(100, 100)):
        canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))

        x_offset = 0
        for i, (image_path, text) in enumerate(zip(images, texts)):
            image = self.resize_image(image_path, image_size)
            canvas.paste(image, (x_offset, 0))
            self.add_text_to_image(canvas, text, position=(x_offset + 5, 5))
            x_offset += image_size[0] + 5

        return canvas

    def get_clip_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def get_prediction(self, frame_path, list_of_labels, how_many_predictions, model, preprocess) -> list:
        Highest3Predictions = []
        try:
            text = clip.tokenize(list_of_labels).to(self.device)
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs = probs.tolist()[0]
            vv = {}
            for i, j in enumerate(probs):
                vv[list_of_labels[i]] = j
            maxx = {k: v for k, v in sorted(vv.items(), key=lambda item: item[1], reverse=True)}
            Highest3Predictions = list(islice(maxx.items(), how_many_predictions))
            print(f"{frame_path} : {Highest3Predictions}")
        except Exception as e:
            print("Exception in CLIP predictions:", e)

        return Highest3Predictions

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

    def startProcess(self, images_path):
        objFD = FaceDetection()
        model, preprocess = self.get_clip_model()
        list_of_age_prompts = clipPrompts.age_prompts
        list_of_gender_prompts = clipPrompts.gender_prompts
        faces_path = []
        detected_age_gender = []
        results = pd.DataFrame(columns=["FrameFileName", "FacePath", "Age Categories", "Gender"])
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
            extract_faces_df = objFD.extractFaces(list_of_images)
            extract_faces_df.to_csv("facesInfo.csv")

            for ind, row in extract_faces_df.iterrows():
                face_path = row['PaddedFacesPath']
                Highest3PredictionsAge = self.get_prediction(face_path, list(list_of_age_prompts.keys()),
                                                             3, model, preprocess)

                Highest3PredictionsGender = self.get_prediction(face_path, list(list_of_gender_prompts.keys()),
                                                                3, model, preprocess)
                Agec1 = Highest3PredictionsAge[0][0]
                Ages1 = round(100 * Highest3PredictionsAge[0][1], 2)

                Genderc1 = Highest3PredictionsGender[0][0]
                Gender1 = round(100 * Highest3PredictionsGender[0][1], 2)
                detected_gender = list_of_gender_prompts[Genderc1]
                detected_age = list_of_age_prompts[Agec1]
                df_length1 = len(results)
                results.loc[df_length1] = [row['FrameFileName'], face_path, detected_age, detected_gender]
                if len(face_path) < 100:
                    faces_path.append(face_path)
                    detected_age_gender.append(detected_age + ":" + detected_gender)
            collage = self.create_collage(faces_path, detected_age_gender)

            # Save the collage
            collage_path = "collageAgeGender.jpg"
            collage.save(collage_path)
            collage.show()
            results.to_csv("AgeGenderResult.csv")
        except Exception as e:
            print(e)
            raise


obj = AgeGenderCategories()
obj.startProcess("images")
