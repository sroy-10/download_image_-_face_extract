import glob
import os.path
import pathlib
import shutil
from os import path

import cv2
import numpy as np
from bing_image_downloader import downloader


def extract_face(src_folder, dest_folder, img_number):
    path = f"dataset/{src_folder}/*"

    # create destination folder
    try:
        os.mkdir(f"dataset/_Extracted/{dest_folder}")
    except:
        pass

    img_file_number = img_number
    img_list = glob.glob(path)

    # Extract face
    for srcfile in img_list:
        try:
            if ".gif" in srcfile:
                continue

            # extract file extension
            file_extension = pathlib.Path(srcfile).suffix

            # read prototxt and caffemodel file
            net = cv2.dnn.readNetFromCaffe(
                "prototxt.txt", "face_detect.caffemodel"
            )

            # load the input image and construct an input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
            image = cv2.imread(srcfile)
            if image is None:
                continue
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300)
            )

            # pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > 0.90:
                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array(
                        [w, h, w, h]
                    )
                    (startX, startY, endX, endY) = box.astype("int")

                    # */ ------------ for debugging purpose ------------ /*
                    print("______________________________________")
                    print(
                        "XY Cordinates: ", (startX, startY, endX, endY)
                    )
                    print("Source File: ", srcfile)
                    print("Image Size: ", image.shape)
                    print("______________________________________")
                    # */ ------------ #end for debugging purpose ------------ /*

                    # Cropping & resizing of the image
                    roi_color = image[startY:endY, startX:endX]
                    resized = cv2.resize(roi_color, (64, 64))

                    # create filename
                    filename = (
                        "dataset/_Extracted/"
                        + dest_folder
                        + "/"
                        + "_".join(dest_folder.split())
                        + "_"
                        + str(img_file_number)
                        + str(file_extension)
                    )
                    img_file_number += 1
                    # storing the image
                    cv2.imwrite(filename, resized)

        except Exception as err:
            print(f"[ERROR] {err=}, {type(err)=}")
            print("source -->", srcfile)
            print("destination ---->", filename)

            # Copy the error file to the Error directory to be processed later
            shutil.copy(srcfile, filename)
            filename = (
                "dataset/_Extracted/_Errors/"
                + "_".join(dest_folder.split())
                + "_"
                + str(img_file_number)
                + str(file_extension)
            )
    return img_file_number


if __name__ == "__main__":
    actor_list = [
        "pankaj tripathi",
        "nawazuddin siddiqui",
        "om puri",
        "amrish puri",
        "anupam kher",
    ]

    for actor in actor_list:
        actor = str(actor).title()

        # download the images
        try:
            downloader.download(
                actor,
                limit=60,
                output_dir="dataset",
                adult_filter_off=True,
                timeout=120,
                filter="photo",
            )
        except:
            pass

        # face extraction
        img_number = 1
        img_number = extract_face(
            src_folder=actor, dest_folder=actor, img_number=img_number
        )

        # remove directory with it content
        shutil.rmtree(f"/dataset/{actor}")
