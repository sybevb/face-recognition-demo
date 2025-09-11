import time
import numpy as np
import cv2
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice
)

INPUT_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.45
CLASSES = ["person", "face"]
COLORS = {
    "person": (255, 0, 0),  # Blue
    "face": (0, 255, 0),  # Green
}


class PersonFaceDetector:
    def __init__(self, model_path):
        print("Start Load Model Person Face Hailo Model ", end='------ ')

        # Hailo Setup
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        self.target = VDevice(params=params)

        self.hef = HEF(model_path)

        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)

        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=True,
                                                             format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False,
                                                               format_type=FormatType.FLOAT32)

        # print("Input vstream infos")
        # for input_vstream_info in self.hef.get_input_vstream_infos():
        #     print("Name:", input_vstream_info.name)
        #     print("Shape:", input_vstream_info.shape)

        # print("Output vstream infos")
        # for output_vstream_info in self.hef.get_output_vstream_infos():
        #     print("Name:", output_vstream_info.name)
        #     print("Shape:", output_vstream_info.shape)

        self.network_group_context = self.network_group.activate(self.network_group_params)
        self.network_group_context.__enter__()  # 🔑 keep network active

        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self.infer_pipeline.__enter__()  # 🔑 keep vstreams alive

        print("Finish!")

    def detect(self, img):
        # img = cv2.imread(image_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)

        outputs = {}

        try:
            infer_results = self.infer_pipeline.infer(input_data)
        except Exception as e:
            print("Inference error:", e)

        conv55 = infer_results["yolov5s_personface/conv55"]
        conv63 = infer_results["yolov5s_personface/conv63"]
        conv70 = infer_results["yolov5s_personface/conv70"]

        outputs = {
            "yolov5s_personface/conv70": conv70,
            "yolov5s_personface/conv63": conv63,
            "yolov5s_personface/conv55": conv55
        }

        detections = self._postprocess(outputs, (h, w))

        return detections

    def _postprocess(self, outputs, img_shape, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
        anchors = [
            [(116, 90), (156, 198), (373, 326)],  # stride 32
            [(30, 61), (62, 45), (59, 119)],  # stride 16
            [(10, 13), (16, 30), (33, 23)]  # stride 8
        ]
        strides = [32, 16, 8]
        layer_names = ["yolov5s_personface/conv70", "yolov5s_personface/conv63", "yolov5s_personface/conv55"]

        num_classes = len(CLASSES)  # person, face
        all_boxes, all_scores, all_classes = [], [], []

        for i, name in enumerate(layer_names):

            pred = np.squeeze(outputs[name], 0)  # (H, W, C)
            h, w, _ = pred.shape
            pred = pred.reshape(h, w, 3, 5 + num_classes)  # 5 box params + num_classes

            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

            for a in range(3):
                tx, ty, tw, th, to = [pred[:, :, a, j] for j in range(5)]
                class_scores = pred[:, :, a, 5:]  # shape (H,W,num_classes)

                bx = (tx * 2 - 0.5 + grid_x) * strides[i]
                by = (ty * 2 - 0.5 + grid_y) * strides[i]
                bw = (tw * 2) ** 2 * anchors[i][a][0]
                bh = (th * 2) ** 2 * anchors[i][a][1]

                x1, y1, x2, y2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2

                # apply sigmoid to objectness and classes
                obj = to
                cls_probs = class_scores

                scores = obj[..., None] * cls_probs  # (H,W,num_classes)
                cls_ids = np.argmax(scores, axis=-1)
                cls_conf = np.max(scores, axis=-1)

                mask = cls_conf > conf_thresh
                if np.any(mask):
                    all_boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=-1))
                    all_scores.append(cls_conf[mask])
                    all_classes.append(cls_ids[mask])

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)

        ho, wo = img_shape
        all_boxes[:, [0, 2]] *= wo / INPUT_SIZE
        all_boxes[:, [1, 3]] *= ho / INPUT_SIZE

        keep = self._nms(all_boxes, all_scores)

        detections = []
        for idx in keep:
            x1, y1, x2, y2 = all_boxes[idx].astype(int)
            score = all_scores[idx]
            cls = int(all_classes[idx])
            detections.append([x1, y1, x2, y2, float(score), cls])

        # detections = filter_face_inside_person(detections)

        return detections

    def _nms(self, boxes, scores, iou_thresh=0.45):
        """Simple greedy NMS"""
        idxs = scores.argsort()[::-1]
        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break

            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            iou = inter / (area_i + area_j - inter)

            idxs = idxs[1:][iou <= iou_thresh]

        return keep

    def close(self):
        if self.infer_pipeline:
            self.infer_pipeline.__exit__(None, None, None)
        if self.network_group_context:
            self.network_group_context.__exit__(None, None, None)
