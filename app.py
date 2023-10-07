import streamlit as st
import cv2
import tempfile
import torch
from matplotlib import pyplot as plt
from PIL import Image

from source.utils import detector_utils, draw_utils
from source import model
from source.model import ContrastiveLoss


device = 'cpu'

# get models
path_to_sn = './models/epoch=39-step=11280.ckpt'
comparison_model = model.get_siamese_model(path_to_sn, device='cpu')

scripted_model = torch.jit.load('./models/detector_scripted.pt', map_location=torch.device(device))
scripted_model = scripted_model.eval().to(device).float()
target_shape = (1280, 720)


DISTANCE_THRESHOLD = 0.27
N_SUSPECIOUS = 3
HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED = 0.05
frame_rate = 0


previous_result = dict()
images_result = dict()



def clear_previous_result(previous_result, n_im=0):
    """Remove """
    rm_set = set()
    for em in previous_result:
        if previous_result[em]['n_frames'][-1] != n_im:
            rm_set.add(em)

    for em in rm_set:
        del previous_result[em]

    return previous_result


def determine_colors(previous_result, n_suspecious=10, color_suspecious='r'):
    for em in previous_result:
        if len(previous_result[em]['n_frames']) >= n_suspecious and previous_result[em]['n_frames'][-1] - previous_result[em]['n_frames'][-n_suspecious] == n_suspecious-1:
            previous_result[em]['last_color'] = color_suspecious
    return previous_result


def object_detection(scripted_model, frame, target_shape, device='cpu'):
    # detect objects on an image
    preprocessed_frame = detector_utils.preprocess_image(frame, target_shape=target_shape)
    res = scripted_model.forward(torch.tensor(preprocessed_frame).to(device).float())
    clone_res = res.clone().detach()
    clone_res_cpu = clone_res.cpu().float()
    clone_res_cpu[:, [0, 1, 2, 5, 6, 7], :, :] = torch.sigmoid(clone_res_cpu[:, [0, 1, 2, 5, 6, 7], :, :])
    nms_thres, iou_threshold = scripted_model.nms_thres, scripted_model.nms_iou_thres
    bboxes = detector_utils.decode_result(clone_res_cpu[0], threshold=nms_thres, iou_threshold=iou_threshold)

    return bboxes


def update_result(previous_result, images_result, bboxes, frame, comparison_model, n_im=0):
    # if there is a detection result in memory, make pairwise comparison of the objects (the current run) with the previous ones
    if previous_result is not None:
        # for detected_object in detection_result:
        for ind, bbox in enumerate(bboxes['boxes']):  # x_min, y_min, x_max, y_max
            if bboxes['labels'][ind] == 0:
                bbox = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], target_shape[0] - 1),
                        min(bbox[3], target_shape[1] - 1)]
                X, Y, W, H = int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
                # if image is too small, don't make the inference
                if (
                    W < target_shape[0] * HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED
                    or H < target_shape[1] * HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED
                ):
                    continue
                # compare (siamese network)
                # if they are similar, save the info in memory (one object to all frames)
                # if they are not, do nothing and go to the next object
                image = Image.fromarray(frame[Y:Y + H, X:X + W], 'RGB')  # cut bbox from the frame
                previous_result, images_result = model.make_inference(comparison_model, model.transforms, image, bbox,
                                                                      previous_result, images_result,
                                                                      n_im=n_im, threshold=DISTANCE_THRESHOLD)
    # replace  detection_result by the new set of objects (current one)
    # check how long we "see" each of objects
    previous_result = clear_previous_result(previous_result, n_im=i)
    # alarm, if the duration is longer than a provided threshold (highlight objects on the video with a bright color)
    previous_result = determine_colors(
        previous_result,
        n_suspecious=N_SUSPECIOUS,
        color_suspecious='r'
    )
    return previous_result, images_result


def draw_resulting_frame(frame, target_shape, previous_result):
    # highlight the found bboxes (suspecious ones are red)
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(cv2.resize(frame, target_shape), cv2.COLOR_BGR2RGB))
    for em in previous_result:
        if previous_result[em]['n_frames'][-1] == i:
            bbox = previous_result[em]['last_bbox']
            color = previous_result[em]['last_color']
            draw_utils.draw_box(bbox, 0, highlight_color_car=color, highlight_color_plate='black')
    plt.axis('off')
    stframe.pyplot(fig)
    plt.close()


if __name__ == '__main__':
    # upload video
    st.title("Video Analysis: Car pursuit identification")
    uploaded_file = st.file_uploader("Upload your video here...", ['mp4', 'mov', 'avi'])
    if uploaded_file:
        # cut video into frames
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        i = 0
        while cap.isOpened():
            # fps of the video equals to 30, thus I add 30 to process one frame per a second
            frame_rate += 30
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_rate)
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

            # detect objects on an image
            bboxes = object_detection(scripted_model, frame, target_shape, device='cpu')

            # if there is a detection result in memory, make pairwise comparison of the objects (the current run) with the previous ones
            previous_result, images_result = update_result(previous_result, images_result, bboxes, frame, comparison_model, n_im=i)

            # highlight the found bboxes (suspecious ones are red)
            draw_resulting_frame(frame, target_shape, previous_result)
            i += 1
