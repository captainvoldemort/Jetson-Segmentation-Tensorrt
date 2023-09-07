import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import random
import ctypes
import pycuda.driver as cuda
import time



EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []


class YoloTRT():
    def __init__(self, library, engine, conf, yolo_ver):
        self.CONF_THRESH = conf 
        self.IOU_THRESHOLD = 0.4
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.yolo_version = yolo_ver
        self.categories = ["bus_stop", "20_mph", "do_not_enter", "do_not_stop", "do_not_turn_l", "do_not_turn_r", "do_not_u_turn", "enter_left_lane", "green_light", "left_right_lane", "no_parking", "parking", "ped_crossing", "ped_zebra_cross", "railway_crossing", "red_light", "stop", "t_intersection_l", "traffic_light", "u_turn", "warning", "yellow_light"]
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def Inference(self, img, out):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]

        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.PostProcess(output[i * self.LEN_ALL_RESULT: (i + 1) * self.LEN_ALL_RESULT], origin_h, origin_w, image_raw, out)

        det_res = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            det = dict()
            det["class"] = self.categories[int(result_classid[j])]
            det["conf"] = result_scores[j]
            det["box"] = box
            det_res.append(det)
            self.PlotBbox(box, img, label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]))

        return det_res, t2 - t1

    def PostProcess(self, output, origin_h, origin_w, image_raw, out):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]

        boxes = self.NonMaxSuppression(pred, origin_h, origin_w, conf_thres=self.CONF_THRESH, nms_thres=self.IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])

        # Process traffic signs (assuming there is only one class for traffic signs)
        class_name = "Traffic Sign"
        for i in range(len(result_boxes)):
            x, y, w, h = result_boxes[i]
            draw_bbox(image_raw, int(x), int(y), int(w), int(h))
            self.process_signs(image_raw, x, y, w, h, out, class_name)

        return result_boxes, result_scores, result_classid

    def process_signs(self, img, x, y, w, h, out, class_name):
        dist_values = []
        area_values = []
        division_values = []

        distance, area, img = self.area_dist(x, y, w, h, img)
        dist_values.append(round(distance, 2))
        area_values.append(round(area, 2))

        div = area_values[0] / dist_values[0]
        division_values.append(round(div, 2))

        cv2.putText(img, f"Class: {class_name}", (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, f"Division: {division_values[0]}", (int(x), int(y) - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        my_dict = {
            "Class": class_name,
            "distance": round(distance, 2),
            "area": round(area, 2),
            "division": round(div, 2)
        }
        print("Traffic Sign - Symbol Detected")
        print(my_dict)
        out.write(img)
            
