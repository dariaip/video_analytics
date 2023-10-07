import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

from utils.trt_utils import init_model, do_inference, allocate_buffers
from utils.detector_utils import preprocess_image, postprocess
from utils.ocr_utils import preprocess_img_ocr, postprocess_ocr
from utils.filter import LPfilter
from utils.draw_utils import box_label, draw_warning, labels_dict


device = 'cuda:0'


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-det', type=str,
                        default='/content/drive/MyDrive/5. Inference/weights/detector.engine',
                        help='model detector path')
    parser.add_argument('--weights-ocr', type=str,
                        default='/content/drive/MyDrive/5. Inference/weights/lpr_epoch_42_ts.engine',
                        help='model ocr path')

    parser.add_argument('--det-imgsz-w', type=int, default=512,
                        help='inference detector size width')
    parser.add_argument('--det-imgsz-h', type=int, default=736,
                        help='inference detector size height')
    parser.add_argument('--ocr-imgsz-w', type=int, default=224, help='inference ocr size width')
    parser.add_argument('--ocr-imgsz-h', type=int, default=224, help='inference ocr size height')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='NMS confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')

    parser.add_argument('--alphabet', type=str, default='0123456789ABCEHKMOPTXY',
                        help='ocr alphabet')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    logger = trt.Logger(trt.Logger.WARNING)
    lp_filter = LPfilter()
    stream = cuda.Stream()

    det_engine, det_model_context, det_model_io = init_model(opt.weights_det, logger)
    ocr_engine, ocr_model_context, ocr_model_io = init_model(opt.weights_ocr, logger)
    det_model_context.active_optimization_profile = 0
    ocr_model_context.active_optimization_profile = 0

    inputs_det, outputs_det, bindings_det, stream, input_shapes_det, out_shapes_det, \
    out_names_det, max_batch_size_det = allocate_buffers(
        det_engine,
        stream,
    )

    inputs_ocr, outputs_ocr, bindings_ocr, stream, input_shapes_ocr, out_shapes_ocr, \
    out_names_ocr, max_batch_size_ocr = allocate_buffers(
        ocr_engine,
        stream,
    )

    cap = cv2.VideoCapture(
      f"uridecodebin uri=file:///content/drive/MyDrive/5.\ Inference/test_images_videos/driving_out_30sec.mp4  ! videoconvert ! video/x-raw, format=BGRx, width=1280, height=720 ! videoconvert ! video/x-raw, format=BGR ! appsink sync=False", 
      cv2.CAP_GSTREAMER
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    cap_write = cv2.VideoWriter(
        f'/content/drive/MyDrive/5.\ Inference/test_images_videos/driving_out_30sec_result.mp4',
        cv2.VideoWriter_fourcc(*'avc1'),
        float(round(fps)),
        (int(736), int(512)),
        # (int(width), int(height)),
    )

    frame_number = 0
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break
        print(frame_number)
        image_raw = cv2.resize(image_np, (736, 512)).copy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        preprocessed_image = preprocess_image(image_np)

        batch_size = preprocessed_image.shape[0]
        allocate_place = np.prod(preprocessed_image.shape)
        inputs_det[0].host[:allocate_place] = preprocessed_image.flatten(order='C').astype(np.float16)
        det_model_context.set_binding_shape(0, preprocessed_image.shape)

        det_outputs = do_inference(
            det_model_context,
            bindings=bindings_det,
            inputs=inputs_det,
            outputs=outputs_det,
            stream=stream
        )[0]

        det_outputs = det_outputs.reshape(tuple(det_model_io['output_shape']))
        bboxes = postprocess(det_outputs, opt.conf_thres, opt.iou_thres)

        for index in range(len(bboxes['boxes'])):
            # if cls is plate
            x1, y1, x2, y2 = [int(cord) for cord in bboxes['boxes'][index]]
            cls = bboxes['labels'][index]
            conf = bboxes['scores'][index]
            text = 'label:' + str(labels_dict[int(cls)]) + \
                   ', x1:' + str(x1) + ', y1:' + str(y1) + \
                   ', x2:' + str(x2) + ', y2:' + str(y2)
            
            text_plate = ''
            time_in_memory = 0
            if int(bboxes['labels'][index]) == 1:
                if  (x2 - x1) > 10 and (y2 - y1) > 5:
                  crop = image_raw[int(y1):int(y2), int(x1):int(x2), :].copy()

                  if crop.size == 0:
                    continue

                  crop = preprocess_img_ocr(crop)

                  batch_size = crop.shape[0]
                  allocate_place = np.prod(crop.shape)
                  inputs_ocr[0].host[:allocate_place] = crop.flatten(order='C').astype(np.float16)
                  ocr_model_context.set_binding_shape(0, crop.shape)
                  trt_outputs_ocr = do_inference(
                      ocr_model_context,
                      bindings=bindings_ocr,
                      inputs=inputs_ocr,
                      outputs=outputs_ocr,
                      stream=stream
                  )
                  trt_outputs_ocr = trt_outputs_ocr[0].reshape(tuple(ocr_model_io['output_shape']))
                  text_plate = postprocess_ocr(trt_outputs_ocr, alphabet=opt.alphabet)

                  first_frame_seen = lp_filter.get_lp_first_seen_frame((x1, y1, x2, y2),
                                                                      text_plate, frame_number)
                                                                      
                  time_in_memory = float((frame_number - first_frame_seen) * time_per_frame)
                  text_plate += f' {time_in_memory:4.1f}'
                  text += 'text: ' + text_plate

            label = f'{labels_dict[int(cls)]} {float(conf):.2f}' + ' ' + text_plate
            color = (0, 0, 255) if time_in_memory > 15 else (255, 100, 0)
            image_raw = box_label(image_raw, (x1, y1, x2, y2), label, color=color)
            if time_in_memory > 15:
                image_raw = draw_warning(image_raw,
                                         warning_text='WARNING: possible persecution')


        lp_filter.clear_history(frame_number)
        frame_number += 1
        cap_write.write(image_raw)

    print('end of video')
