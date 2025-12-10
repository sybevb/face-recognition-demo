import argparse
import cairo
import cv2
import numpy as np
import os

from detection.personface_detection import PersonFaceDetector, CLASSES, COLORS
# from detection.emotion_detection import EmotionDetector

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


class PFDemo:
    def __init__(self, video_device, inf_device, model_path):
        self.inited = False
        self.detections = []
        self.emo = ""

        if os.path.exists("/usr/lib/libvx_delegate.so"):
            self.platform = "i.MX8MP"
            print("Platform i.MX8MP detected")
        elif os.path.exists("/usr/lib/libethosu_delegate.so"):
            self.platform = "i.MX93"
            print("Platform i.MX93 detected")
        else:
            print("Unknown platform")
            pass

        cam_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,framerate=60/1,width=640,height=640,format=YUY2 ! "
            "videoconvert ! "
            "tee name=t "
            "t. ! queue max-size-buffers=2 leaky=2 ! cairooverlay name=drawer ! autovideosink sync=false "
            "t. ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink emit-signals=true drop=true max-buffers=2 name=ml_sink"
        )

        pipeline = Gst.parse_launch(cam_pipeline)
        pipeline.set_state(Gst.State.PLAYING)

        drawer = pipeline.get_by_name("drawer")
        drawer.connect("draw", self.draw)

        ml_sink = pipeline.get_by_name("ml_sink")
        ml_sink.connect("new-sample", self.inference)

        # face_model = model_path + "/face_detection_ptq.tflite" # using nxp imx8mp
        pf_model = model_path + "/yolov5s_personface.hef"
        # emo_model = model_path + "/emotion_uint8_float32.tflite"

        if self.platform == "i.MX93" and inf_device == "NPU":
            face_model = model_path + "/face_detection_ptq_vela.tflite"

        self.pf_detector = PersonFaceDetector(pf_model)

        # self.emo_detector = EmotionDetector(emo_model, inf_device, self.platform)

        self.inited = True

    def inference(self, data):
        detections = []
        # emo = ""

        frame = data.emit("pull-sample")

        buf = frame.get_buffer()
        caps = frame.get_caps()
        h = caps.get_structure(0).get_value("height")
        w = caps.get_structure(0).get_value("width")

        # Map GstBuffer → numpy array
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
        buf.unmap(map_info)

        detections = self.pf_detector.detect(frame)

        # if np.size(detections, 0) > 0:
        #     self.detections = detections  # save for overlay

        #     for x1, y1, x2, y2, score, cls in detections:
        #         if cls == 1:
        #             face_crop = frame[y1:y2, x1:x2]
        #             try:
        #                 emo = self.emo_detector.detect(face_crop)
        #                 self.emo = emo
        #             except Exception as e:
        #                 pass
        # else:
        #     self.detections = detections
        #     self.emo = emo
        self.detections = detections  # save for overlay

        return Gst.FlowReturn.OK

    def draw(self, overlay, context, timestamp, duration):
        # Set Cairo properties
        context.set_line_width(2)

        for x1, y1, x2, y2, score, cls in self.detections:
            if cls == 1:
                label = f"{CLASSES[cls]} {score:.2f} {self.emo}"
            else:
                label = f"{CLASSES[cls]} {score:.2f}"

            r, g, b = [c / 255.0 for c in COLORS[CLASSES[cls]]]

            # Draw rectangle
            context.set_source_rgb(r, g, b)
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()

            # Draw label
            context.move_to(x1, y1 - 5)
            context.set_font_size(16)
            context.show_text(label)

    def close(self):
        self.pf_detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="/dev/video0", help="Camera device to be used"
    )
    parser.add_argument(
        "--backend", type=str, default="NPU", help="Use NPU or CPU to do inference"
    )
    parser.add_argument(
        "--model_path", type=str, default="models", help="Path for models and image"
    )
    args = parser.parse_args()

    Gst.init(None)
    demo = PFDemo(args.device, args.backend, args.model_path)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Pipeline stopped by user")
    finally:
        demo.close()
        loop.quit()