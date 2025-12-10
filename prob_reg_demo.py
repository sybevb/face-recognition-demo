import argparse
import cairo
import cv2
import numpy as np
import os
import time

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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


# -----------------------------
# Hailo model wrapper
# -----------------------------

INPUT_SIZE = 224  # your model input: 3 x 224 x 224
CLASS_NAMES = ["Pipe", "No View"]  # change if you have specific names


class ProbRegModel:
    """
    Simple wrapper around a Hailo HEF that outputs:
      - cls_logits: [1, 2]
      - regression: [1, 3]
    """

    def __init__(self, hef_path: str):
        print("Loading Hailo model:", hef_path, "------ ", end="")

        # Hailo setup
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        self.target = VDevice(params=params)

        self.hef = HEF(hef_path)

        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )

        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Input: quantized uint8; layout is defined in the HEF
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group,
            quantized=True,
            format_type=FormatType.UINT8
        )

        # Outputs: dequantized float32
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        # Activate network group + vstreams
        self.network_group_context = self.network_group.activate(self.network_group_params)
        self.network_group_context.__enter__()  # keep network active

        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params
        )
        self.infer_pipeline.__enter__()  # keep vstreams alive

        print("Done!")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess frame to match input: [N, 3, 224, 224], uint8.
        """
        img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        img_chw = np.transpose(img_rgb, (2, 0, 1))  # (3, 224, 224)

        # Add batch dim → (1, 3, 224, 224)
        input_data = np.expand_dims(img_chw, axis=0).astype(np.uint8)
        return input_data

    def infer(self, frame_bgr: np.ndarray):
        """
        Run inference on a single frame.

        Returns:
            cls_probs: np.ndarray of shape (2,)
            regression: np.ndarray of shape (3,)
        """
        input_data = self._preprocess(frame_bgr)

        try:
            infer_results = self.infer_pipeline.infer(input_data)
        except Exception as e:
            print("Inference error:", e)
            # Return zeros so overlay stays valid
            return np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # Expecting these names in the HEF output vstreams
        cls_logits = infer_results["cls_logits"]  # shape [1, 2]
        regression = infer_results["regression"]  # shape [1, 3]

        cls_logits = np.squeeze(cls_logits, axis=0)  # (2,)
        regression = np.squeeze(regression, axis=0)  # (3,)

        # Softmax to get probabilities
        e = np.exp(cls_logits - np.max(cls_logits))
        cls_probs = e / np.sum(e)

        return cls_probs, regression

    def close(self):
        if hasattr(self, "infer_pipeline") and self.infer_pipeline:
            self.infer_pipeline.__exit__(None, None, None)
        if hasattr(self, "network_group_context") and self.network_group_context:
            self.network_group_context.__exit__(None, None, None)


# -----------------------------
# GStreamer demo with overlay
# -----------------------------

class ProbRegDemo:
    def __init__(self, video_device: str, model_path: str):
        self.inited = False

        # Values used by the overlay
        self.cls_probs = np.zeros(2, dtype=np.float32)
        self.regression = np.zeros(3, dtype=np.float32)

        # Detect platform (kept from original example, but not strictly needed)
        if os.path.exists("/usr/lib/libvx_delegate.so"):
            self.platform = "i.MX8MP"
            print("Platform i.MX8MP detected")
        elif os.path.exists("/usr/lib/libethosu_delegate.so"):
            self.platform = "i.MX93"
            print("Platform i.MX93 detected")
        else:
            self.platform = "Unknown"
            print("Unknown platform")

        # Camera pipeline (use provided device)
        cam_pipeline = (
            f"v4l2src device={video_device} ! "
            "video/x-raw,framerate=60/1,width=640,height=640,format=YUY2 ! "
            "videoconvert ! "
            "tee name=t "
            "t. ! queue max-size-buffers=2 leaky=2 ! cairooverlay name=drawer ! autovideosink sync=false "
            "t. ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink emit-signals=true drop=true max-buffers=2 name=ml_sink"
        )

        pipeline = Gst.parse_launch(cam_pipeline)
        self.pipeline = pipeline
        pipeline.set_state(Gst.State.PLAYING)

        drawer = pipeline.get_by_name("drawer")
        drawer.connect("draw", self.draw)

        ml_sink = pipeline.get_by_name("ml_sink")
        ml_sink.connect("new-sample", self.inference)

        # ---- Hailo model ----
        # Change file name here to your HEF
        hef_path = os.path.join(model_path, "model_fused_fixed.hef")
        self.model = ProbRegModel(hef_path)

        self.inited = True

    def inference(self, data):
        """
        appsink callback: pull frame, run inference, store results.
        """
        frame_sample = data.emit("pull-sample")
        buf = frame_sample.get_buffer()
        caps = frame_sample.get_caps()
        h = caps.get_structure(0).get_value("height")
        w = caps.get_structure(0).get_value("width")

        # Map GstBuffer → numpy array
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
        buf.unmap(map_info)

        # Run Hailo inference on BGR frame
        cls_probs, regression = self.model.infer(frame)

        # Save for overlay
        self.cls_probs = cls_probs
        self.regression = regression

        return Gst.FlowReturn.OK

    def draw(self, overlay, context: cairo.Context, timestamp, duration):
        """
        cairooverlay callback: draw a box at top-left with
        class probabilities and regression values.
        """
        # Box position and size
        margin_x = 10
        margin_y = 10
        box_width = 320
        box_height = 110

        # Background rectangle (semi-transparent black)
        try:
            context.set_source_rgba(0, 0, 0, 0.6)
        except TypeError:
            # Fallback if rgba is not supported; just solid black
            context.set_source_rgb(0, 0, 0)
        context.rectangle(margin_x, margin_y, box_width, box_height)
        context.fill()

        # Text color: white
        context.set_source_rgb(1, 1, 1)
        context.set_font_size(18)

        x_text = margin_x + 10
        y_text = margin_y + 25

        # Line 1: header
        context.move_to(x_text, y_text)
        context.show_text("Model output:")

        # Line 2 & 3: probabilities
        y_text += 22
        context.move_to(x_text, y_text)
        context.show_text(
            f"{CLASS_NAMES[0]}: {self.cls_probs[0]:.3f}"
        )

        y_text += 22
        context.move_to(x_text, y_text)
        context.show_text(
            f"{CLASS_NAMES[1]}: {self.cls_probs[1]:.3f}"
        )

        # Line 4: regression
        y_text += 22
        r0, r1, r2 = self.regression
        context.move_to(x_text, y_text)
        context.show_text(
            f"reg: [{r0:.3f}, {r1:.3f}, {r2:.3f}]"
        )

    def close(self):
        if hasattr(self, "model") and self.model:
            self.model.close()
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)


# -----------------------------
# main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="/dev/video0",
        help="Camera device to be used"
    )
    parser.add_argument(
        "--backend", type=str, default="NPU",
        help="(Ignored) kept for compatibility with original script"
    )
    parser.add_argument(
        "--model_path", type=str, default="models",
        help="Path containing custom_model.hef"
    )
    args = parser.parse_args()

    Gst.init(None)
    demo = ProbRegDemo(args.device, args.model_path)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Pipeline stopped by user")
    finally:
        demo.close()
        loop.quit()
