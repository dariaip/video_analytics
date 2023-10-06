import streamlit as st
import cv2
import tempfile
import torch
import torchvision
import pytorch_lightning as pl
from source.utils import detector_utils, draw_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


previous_result = dict()
images_result = dict()

DISTANCE_THRESHOLD = 0.27
N_SUSPECIOUS = 3
im_size = (224,224)
frame_rate = 0
HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED = 0.05

#model = torchvision.models.resnet18(pretrained=True)
#model.fc = torch.nn.Linear(in_features=512, out_features=512)
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(im_size),
    torchvision.transforms.ToTensor(),
])

class SiameseNetwork(pl.LightningModule):
    def __init__(self, criterion, embeddingNet):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.criterion = criterion
        self.all_embs = []
        self.all_lbls = []

        self.embeddingNet = embeddingNet

    def forward_once(self, x):
        return self.embeddingNet.forward(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        im1, im2, label = batch
        out1, out2 = self.forward(im1, im2)
        loss = self.criterion(out1, out2, label)
        return loss

    def validation_step(self, batch, batch_idx):
        self._collect_embeddings(batch, batch_idx)

    def on_validation_epoch_end(self):
        dist_ratio = self._calculate_map('val')
        self.all_embs = []
        self.all_lbls = []
        return dist_ratio

    def test_step(self, batch, batch_idx):
        self._collect_embeddings(batch, batch_idx)

    def on_test_epoch_end(self):
        dist_ratio = self._calculate_map('test')
        self.all_embs = []
        self.all_lbls = []
        return dist_ratio

    def _collect_embeddings(self, batch, batch_idx):
        # collect all embeddings and related labels for the batch on a validation step
        self.embeddingNet.eval()
        with torch.no_grad():
            imgs, lbls = batch
            embs = self.forward_once(imgs.to(device)).detach().cpu().numpy()
            self.all_embs += [e for e in embs]
            self.all_lbls += lbls

    def _calculate_map(self, step_label):
        # when validation is over, calculate mAP metrics for the collected embeddings
        aps = []
        for lbl_idx in range(len(self.all_lbls)):
            all_embs_tmp = self.all_embs.copy()
            all_lbls_tmp = self.all_lbls.copy()
            ancor_emb = all_embs_tmp.pop(lbl_idx)
            ancor_lbl = all_lbls_tmp.pop(lbl_idx)
            distances = np.linalg.norm(np.array(all_embs_tmp) - ancor_emb, axis=1)
            distances, all_lbls_tmp = (list(t) for t in zip(*sorted(zip(distances, all_lbls_tmp))))
            found_cnt = 0
            ap = 0
            for i, lbl in enumerate(all_lbls_tmp):
                if lbl == ancor_lbl:
                    found_cnt += 1
                    ap += found_cnt / (i+1)
            ap /= found_cnt
            aps.append(ap)

        mAP = sum(aps)/len(aps)
        self.log(f'{step_label}_map', mAP)
        return mAP

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.embeddingNet.parameters(), lr=1e-3)
        return optimizer

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance
        dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label)*torch.pow(dist, 2) + label*torch.clamp(self.margin - torch.pow(dist, 2), min=0)) # YOUR CODE HERE
        return loss_contrastive

path_to_sn = './models/epoch=39-step=11280.ckpt'
model = SiameseNetwork.load_from_checkpoint(path_to_sn, map_location=torch.device('cpu'))


def make_inference(model, transforms, image, bbox, previous_result, n_im=0, threshold=0.8, color_basic='b'):
    """Making inference for a detected car, as well as compating it with previously seen cars"""
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = transforms(image).to(device)
        x = torch.unsqueeze(x, dim=0)
        em = model.forward_once(x).detach().cpu().numpy()[0]
        current_results = dict()
        for n_em_prev in previous_result:
            dist = np.linalg.norm(em - images_result[n_em_prev], axis=0)
            if dist < threshold:
                current_results[n_em_prev] = dist

        if not current_results or n_im == 0:
            if previous_result:
                n_em_prev = max(previous_result.keys()) + 1
            else:
                n_em_prev = 0
            previous_result[n_em_prev] = {'n_frames': [n_im], 'last_bbox': bbox, 'dist': threshold, 'last_color': color_basic}
            images_result[n_em_prev] = em
        else:
            for n_em_prev, dist in sorted(current_results.items(), key=lambda x: x[1]):
                if previous_result[n_em_prev]['n_frames'][-1] != n_im:
                    previous_result[n_em_prev]['n_frames'].append(n_im)
                    previous_result[n_em_prev]['last_bbox'] = bbox
                    previous_result[n_em_prev]['dist'] = current_results[n_em_prev]
                    images_result[n_em_prev] = em
                    break

        return previous_result


def clear_previous_result(previous_result, n_suspecious=10, n_im=0):
    """Remove """
    rm_set = set()
    for em in previous_result:
        if previous_result[em]['n_frames'][-1] != n_im: #or (
            #len(previous_result[em]['n_frames']) > n_suspecious
            #and n_im - previous_result[em]['n_frames'][-n_suspecious] > n_suspecious
            #) or n_im - previous_result[em]['n_frames'][-1] > n_suspecious:
            rm_set.add(em)

    for em in rm_set:
        del previous_result[em]

    return previous_result


def determine_colors(previous_result, n_suspecious=10, n_im=0, color_suspecious='r'):
    for em in previous_result:
        if len(previous_result[em]['n_frames']) >= n_suspecious and previous_result[em]['n_frames'][-1] - previous_result[em]['n_frames'][-n_suspecious] == n_suspecious-1:
            previous_result[em]['last_color'] = color_suspecious
    return previous_result


# upload video
st.title("Demonstration of the Video Analysis pipeline")
uploaded_file = st.file_uploader("Upload your video here...", ['mp4', 'mov', 'avi'])
if uploaded_file:
    # st.video(uploaded_file)

    # cut video into images
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    i = 0
    while cap.isOpened():
        frame_rate += 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_rate)
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        #stframe.image(frame)

        # detect objects on the image
        device = 'cpu'
        scripted_model = torch.jit.load(
            './models/detector_scripted.pt',
            map_location=torch.device(device)
        )
        scripted_model = scripted_model.eval().to(device).float()
        target_shape = (1280, 720) #(736, 512)
        preprocessed_frame = detector_utils.preprocess_image(frame, target_shape=target_shape)
        res = scripted_model.forward(torch.tensor(preprocessed_frame).to(device).float())
        clone_res = res.clone().detach()
        clone_res_cpu = clone_res.cpu().float()
        clone_res_cpu[:, [0, 1, 2, 5, 6, 7], :, :] = torch.sigmoid(clone_res_cpu[:, [0, 1, 2, 5, 6, 7], :, :])
        nms_thres, iou_threshold = scripted_model.nms_thres, scripted_model.nms_iou_thres
        bboxes = detector_utils.decode_result(clone_res_cpu[0], threshold=nms_thres, iou_threshold=iou_threshold)
        # highlight objects on the video with a shadowed color
        #fig = plt.figure()
        #plt.imshow(cv2.resize(frame, target_shape))
        #for index in range(len(bboxes['boxes'])):
        #    if bboxes['labels'][index] == 0:
        #        draw_utils.draw_box(bboxes['boxes'][index], bboxes['labels'][index],
        #                            highlight_color_car='g', highlight_color_plate='r')
        #stframe.pyplot(fig)

        # if there is a detection result in memory, make pairwise comparison of the objects (the current run) with the previous ones
        if previous_result is not None:
            # for detected_object in detection_result:
            for ind, bbox in enumerate(bboxes['boxes']): # x_min, y_min, x_max, y_max
                if bboxes['labels'][ind] == 0:
                    bbox = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], target_shape[0]-1), min(bbox[3], target_shape[1]-1)]
                    X, Y, W, H = int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1])
                    # if image is too small, don't make the inference
                    if W < target_shape[0] * HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED or H < target_shape[1] * HOW_SMALL_BBOX_SHOULD_BE_CONSIDERED:
                        continue
                    # compare (siam networks?)
                    # if they are similar, save the info in memory (one object to all frames)
                    # if they are not, do nothing and go to the next object
                    image = Image.fromarray(frame[Y:Y+H, X:X+W], 'RGB') # cut bbox from the frame
                    previous_result = make_inference(model, transforms, image, bbox, previous_result,
                                                     n_im=i, threshold=DISTANCE_THRESHOLD)
        # replace  detection_result by the new set of objects (current one)
        # check how long we "see" each of objects
        previous_result = clear_previous_result(previous_result, n_suspecious=N_SUSPECIOUS, n_im=i)
        # alarm, if the duration is longer than a provided threshold (highlight objects on the video with a bright color)
        previous_result = determine_colors(
            previous_result,
            n_suspecious=N_SUSPECIOUS, n_im=i,
            color_suspecious='r'
        )
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
        i += 1
        print()
        print(len(previous_result))
        for em in previous_result:
            print(em, len(previous_result[em]['n_frames']), previous_result[em]['last_bbox'], previous_result[em]['dist'], previous_result[em]['last_color'])
