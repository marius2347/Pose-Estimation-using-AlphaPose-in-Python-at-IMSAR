# live camera with boxes detection of humans

import cv2
import os
import mxnet as mx
from gluoncv import model_zoo, data, utils
import numpy as np
import matplotlib.pyplot as plt

# load the models
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

# detect only persons
detector.reset_class(["person"], reuse_weights=['person'])

def detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs):
    pose_input = []
    upscale_bbox = []
    for i in range(len(class_IDs[0])):
        if scores[0][i] > 0.5:
            if class_IDs[0][i].asscalar() == 0:  # class 'person' is labeled as 0
                bbox = bounding_boxs[0][i].asnumpy()
                x_min, y_min, x_max, y_max = bbox
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # ensure the coordinates are within the image bounds
                h, w, _ = img.shape
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                # check if the bounding box is valid
                if x_min >= x_max or y_min >= y_max:
                    continue
                
                person_img = img[y_min:y_max, x_min:x_max]
                
                if person_img.size == 0:
                    print(f"Warning: Extracted person image is empty at bbox {bbox}")
                    continue
                
                person_img = cv2.resize(person_img, (192, 256))
                person_img = mx.nd.array(person_img).astype('float32')
                person_img = mx.nd.image.to_tensor(person_img)
                person_img = mx.nd.image.normalize(person_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                person_img = person_img.expand_dims(axis=0)
                pose_input.append(person_img)
                upscale_bbox.append((x_min, y_min, x_max, y_max))
    if pose_input:
        pose_input = mx.nd.concat(*pose_input, dim=0)
    return pose_input, upscale_bbox

def get_max_pred(heatmap):
    coords = []
    confidence = []
    for i in range(heatmap.shape[0]):
        heat = heatmap[i].asnumpy()
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        conf = heat[y, x]
        coords.append([x, y])
        confidence.append(conf)
    return np.array(coords), np.array(confidence)

def heatmap_to_coord_alpha_pose(heatmap, upscale_bbox):
    pred_coords = []
    confidence = []
    for i in range(len(upscale_bbox)):
        coords, conf = get_max_pred(heatmap[i])
        pred_coords.append(coords)
        confidence.append(conf)
    pred_coords = np.array(pred_coords)
    confidence = np.array(confidence)

    # debugging: Print shapes
    print(f"Predicted coords shape: {pred_coords.shape}")
    print(f"Confidence shape: {confidence.shape}")

    if confidence.ndim == 1:
        confidence = np.expand_dims(confidence, axis=1)  # Make it 2D if needed

    return pred_coords, confidence

def plot_keypoints_v2(img, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2):
    img = np.asarray(img).copy()
    for i, (coords, conf) in enumerate(zip(pred_coords, confidence)):
        if scores[0][i] > box_thresh:
            # draw bounding box
            bbox = bounding_boxs[0][i].asnumpy()
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            for j, (coord, conf_value) in enumerate(zip(coords, conf)):
                if conf_value > keypoint_thresh:
                    x, y = coord
                    cv2.circle(img, (int(x + x_min), int(y + y_min)), 5, (0, 255, 0), -1)
    return img

# Start video capture
cap = cv2.VideoCapture(4)  

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    x, img = data.transforms.presets.yolo.transform_test(mx.nd.array(frame), short=512)

    # detect persons in the frame
    class_IDs, scores, bounding_boxs = detector(x)

    if isinstance(x, mx.nd.NDArray):
        pose_input, upscale_bbox = detector_to_alpha_pose(frame, class_IDs, scores, bounding_boxs)
        if len(pose_input) > 0:
            predicted_heatmap = pose_net(pose_input)
            pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

            try:
                img_with_keypoints = plot_keypoints_v2(frame, pred_coords, confidence,
                                                       class_IDs, bounding_boxs, scores,
                                                       box_thresh=0.5, keypoint_thresh=0.2)
                cv2.imshow('Keypoints', img_with_keypoints)
            except Exception as e:
                print(f"Visualization error: {e}")

    cv2.imshow('Frame', frame)

    # break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
