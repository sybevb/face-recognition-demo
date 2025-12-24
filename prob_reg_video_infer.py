"""
Probabilistic classification + regression inference on an MP4 file using Hailo.

Outputs a CSV log in a local logs/ folder.
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


INPUT_SIZE = 224  # from HEF: (224, 224, 3)
CLASS_NAMES = ["No view", "Pipe"]
REGRESSION_NAMES = ["Spray", "Under water", "Dirty lens"]


class ProbRegModel:
    """
    Wrapper around Hailo HEF (models/my_tflite_model.hef).
    """

    def __init__(self, hef_path: str):
        print("Loading Hailo model:", hef_path, "------ ", end="")

        self.input_name = "my_tflite_model/input_layer1"
        self.cls_name = "my_tflite_model/softmax1"
        self.reg_name = "my_tflite_model/fc4"

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

        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group,
            quantized=True,
            format_type=FormatType.UINT8,
        )

        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        self.network_group_context = self.network_group.activate(self.network_group_params)
        self.network_group_context.__enter__()

        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        )
        self.infer_pipeline.__enter__()

        print("Done!")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        return input_data

    def infer(self, frame_bgr: np.ndarray):
        input_data = self._preprocess(frame_bgr)

        try:
            infer_results = self.infer_pipeline.infer(input_data)
        except Exception as e:
            print("Inference error:", e)
            return np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)

        if self.cls_name not in infer_results or self.reg_name not in infer_results:
            print("Available output keys:", list(infer_results.keys()))
            raise KeyError(
                f"Expected outputs '{self.cls_name}' and '{self.reg_name}' not found in infer_results."
            )

        cls_probs = np.array(infer_results[self.cls_name]).reshape(-1)
        if cls_probs.size < 2:
            tmp = np.zeros(2, dtype=np.float32)
            tmp[:cls_probs.size] = cls_probs
            cls_probs = tmp
        elif cls_probs.size > 2:
            cls_probs = cls_probs[:2]

        s = float(np.sum(cls_probs))
        if s > 0:
            cls_probs = cls_probs / s

        regression = np.array(infer_results[self.reg_name]).reshape(-1)
        if regression.size < 3:
            tmp = np.zeros(3, dtype=np.float32)
            tmp[:regression.size] = regression
            regression = tmp
        elif regression.size > 3:
            regression = regression[:3]

        return cls_probs.astype(np.float32), regression.astype(np.float32)

    def close(self):
        if hasattr(self, "infer_pipeline") and self.infer_pipeline:
            self.infer_pipeline.__exit__(None, None, None)
        if hasattr(self, "network_group_context") and self.network_group_context:
            self.network_group_context.__exit__(None, None, None)


class ProbRegVideoInfer:
    def __init__(self, input_path: str, model_path: str, log_dir: str):
        self.inited = False
        self.frame_idx = 0
        self.loop = None

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"prob_reg_{ts}.csv")
        self.log_f = open(log_path, "w", encoding="ascii")
        self.log_f.write(
            "frame_idx,timestamp_sec,cls_0,cls_1,reg_0,reg_1,reg_2\n"
        )
        self.log_f.flush()

        hef_path = os.path.join(model_path, "my_tflite_model.hef")
        self.model = ProbRegModel(hef_path)

        pipeline_desc = (
            f"filesrc location=\"{input_path}\" ! "
            "decodebin ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=ml_sink emit-signals=true sync=false max-buffers=1 drop=false"
        )

        self.pipeline = Gst.parse_launch(pipeline_desc)

        ml_sink = self.pipeline.get_by_name("ml_sink")
        ml_sink.connect("new-sample", self.inference)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline started. Logging to:", log_path)

        self.inited = True

    def inference(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        h = caps.get_structure(0).get_value("height")
        w = caps.get_structure(0).get_value("width")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
        buf.unmap(map_info)

        cls_probs, regression = self.model.infer(frame)

        pts = buf.pts
        if pts == Gst.CLOCK_TIME_NONE:
            ts_sec = time.time()
        else:
            ts_sec = pts / Gst.SECOND

        self.log_f.write(
            f"{self.frame_idx},{ts_sec:.6f},"
            f"{cls_probs[0]:.6f},{cls_probs[1]:.6f},"
            f"{regression[0]:.6f},{regression[1]:.6f},{regression[2]:.6f}\n"
        )
        self.log_f.flush()

        self.frame_idx += 1
        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print("End of stream")
            self.stop()
        elif msg_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print("Pipeline error:", err)
            if dbg:
                print("Debug info:", dbg)
            self.stop()

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.model:
            self.model.close()
        if self.log_f:
            self.log_f.close()
        if self.loop:
            self.loop.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str, help="Path to local mp4 file"
    )
    parser.add_argument(
        "--model_path", type=str, default="models", help="Path containing my_tflite_model.hef"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to write logs"
    )
    args = parser.parse_args()

    Gst.init(None)
    runner = ProbRegVideoInfer(args.input, args.model_path, args.log_dir)

    loop = GLib.MainLoop()
    runner.loop = loop
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
