import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class EmotionDetector:
    def __init__(self, model_path, inf_device, platform):
        if inf_device == "NPU":
            if platform == "i.MX8MP":
                delegate = tflite.load_delegate("/usr/lib/libvx_delegate.so")
            elif platform == "i.MX93":
                delegate = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
            else:
                print("Platform not supported!")
                return
            self.interpreter = tflite.Interpreter(
                model_path=model_path, experimental_delegates=[delegate]
            )
        else:
            # inf_device is CPU
            self.interpreter = tflite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()

        # model warm up
        time_start = time.time()
        self.interpreter.invoke()
        time_end = time.time()
        print("Emotion detection warm up time:")
        print((time_end - time_start) * 1000, " ms")

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.quantization = self.input_details[0]["quantization"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def _pre_processing(self, input_data):
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        input_data = cv2.resize(input_data, (48, 48))
        input_data = input_data[None, ..., None] / 255.0

        input_scale, input_zero_point = self.quantization
        input_data = input_data / input_scale + input_zero_point

        input_data = input_data.astype(np.uint8)
        return input_data

    def detect(self, img):
        input_data = self._pre_processing(img)
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        emo = LABELS[out.argmax()]
        # print("Recognized emotion:", emo)
        return emo

    def benchmark(self, img):
        n_runs = 50
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.detect(img)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results = {
            "avg_latency_ms": np.mean(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "throughput_fps": 1000.0 / np.mean(times),
        }
        return results