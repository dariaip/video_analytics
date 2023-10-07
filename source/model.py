import torchvision
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np


im_size = (224,224)


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

    def _collect_embeddings(self, batch, batch_idx, device='cpu'):
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
        loss_contrastive = torch.mean((1-label)*torch.pow(dist, 2) + label*torch.clamp(self.margin - torch.pow(dist, 2), min=0))
        return loss_contrastive


def get_siamese_model(path, device='cpu'):
    return SiameseNetwork.load_from_checkpoint(path, map_location=torch.device(device))


def make_inference(model, transforms, image, bbox, previous_result, images_result,
                   n_im=0, threshold=0.8, color_basic='b', device='cpu'):
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

        return previous_result, images_result