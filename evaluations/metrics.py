from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
import monai
import numpy as np
import torch
from monai.metrics.metric import CumulativeIterationMetric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity, BinaryConfusionMatrix

bone_name_dict = {0: "Capitate", 1: "DistalRadius", 2: "DistalUlna", 3: "Hamate", 4: "Lunate", 5: "Pisifrom&Triquetrum",
                  6: "Scaphoid", 7: "Trapzium", 8: "Trapzoid", 9: "metacarpal1st", 10: "metacarpal2nd",
                  11: "metacarpal3rd", 12: "metacarpal4th", 13: "metacarpal5th"}


class RAVDMetric(CumulativeIterationMetric):
    """
    MONAI-compatible RAVD metric (Relative Absolute Volume Difference).
    """

    def __init__(self, include_background: bool = False):
        super().__init__()
        self.include_background = include_background

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        B, C = y_pred.shape[:2]
        results = []

        for b in range(B):
            sample_result = []
            for c in range(C):
                if not self.include_background and c == 0:
                    continue
                pred_bin = (y_pred[b, c] > 0.5).astype(np.uint8)
                gt_bin = (y[b, c] > 0.5).astype(np.uint8)
                gt_vol = int(gt_bin.sum())
                if gt_vol == 0:
                    val = 0.0
                else:
                    pred_vol = int(pred_bin.sum())
                    val = abs(pred_vol - gt_vol) / (gt_vol + 1e-8)
                sample_result.append(val)
            results.append(sample_result)
        return torch.tensor(results)

    def reset(self) -> None:
        super().reset()


class VOEMetric:
    def __init__(self, iou_func):
        self.iou = iou_func

    def __call__(self, pred_bin, gt_bin):
        assert pred_bin.shape == gt_bin.shape, "Shape mismatch between pred and gt"
        iou = self.iou(pred_bin, gt_bin)
        voe = 1 - iou
        return voe


class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_labels = num_classes
        self.fnames = []
        # self.acc_per_channel = []
        # self.acc_reduced = []
        # self.prec_per_channel = []
        # self.prec_reduced = []
        # self.recall_per_channel = []
        # self.recall_reduced = []
        # self.f1_per_channel = []
        # self.f1_reduced = []
        self.dsc_per_channel = []
        self.dsc_reduced = []
        self.nsd_per_channel = []
        self.nsd_reduced = []
        # self.hd95_per_channel = []
        # self.hd95_reduced = []
        self.voe_per_channel = []
        self.voe_reduced = []
        self.msd_per_channel = []
        self.msd_reduced = []
        self.ravd_per_channel = []
        self.ravd_reduced = []
        # self.accuracy = MultilabelAccuracy(num_labels=num_labels, average="none")
        # self.precision = MultilabelPrecision(num_labels=num_labels, average="none")
        # self.recall = MultilabelRecall(num_labels=num_labels, average="none")
        # self.f1 = MultilabelF1Score(num_labels=num_labels, average="none")
        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(class_thresholds=[2 for _ in range(num_classes)],include_background=True, reduction="none")
        # self.nsd = monai.metrics.SurfaceDistanceMetric(include_background=True, reduction="none") # compute_surface_dice
        # self.hd95 = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none")
        self.ravd = RAVDMetric(include_background=True)

    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)
        pred_bin = pred_bin.detach().cpu()
        gt = gt.detach().cpu()
        # pred_flat = pred_bin.permute(0, 2, 3, 1).reshape(-1, pred_bin.shape[1]).detach().cpu()
        # gt_flat = gt.permute(0, 2, 3, 1).reshape(-1, gt.shape[1]).detach().cpu()

        # acc_pc = self.accuracy(pred_flat, gt_flat)
        # prec_pc = self.precision(pred_flat, gt_flat)
        # recall_pc = self.recall(pred_flat, gt_flat)
        # f1_pc = self.f1(pred_flat, gt_flat)

        dsc_pc = self.dsc(pred_bin, gt).squeeze()
        nsd_pc = self.nsd(pred_bin, gt).squeeze()
        voe_pc = self.voe(pred_bin, gt).squeeze()
        msd_pc = self.msd(pred_bin, gt).squeeze()
        ravd_pc = self.ravd(pred_bin, gt).squeeze()
        # hd95_pc = self.hd95(pred_bin, gt).squeeze()

        # self.acc_per_channel.append(acc_pc)
        # self.prec_per_channel.append(prec_pc)
        # self.recall_per_channel.append(recall_pc)
        # self.f1_per_channel.append(f1_pc)

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)
        # self.hd95_per_channel.append(hd95_pc)

        # self.acc_reduced.append(acc_pc.mean())
        # self.prec_reduced.append(prec_pc.mean())
        # self.recall_reduced.append(recall_pc.mean())
        # self.f1_reduced.append(f1_pc.mean())

        self.dsc_reduced.append(dsc_pc.mean())
        self.nsd_reduced.append(nsd_pc.mean())
        self.voe_reduced.append(voe_pc.mean())
        self.msd_reduced.append(msd_pc.mean())
        self.ravd_reduced.append(ravd_pc.mean())
        # self.hd95_reduced.append(hd95_pc.mean())

    def get_metrics(self):
        metrics = {
            # "accuracy_pc": np.array(self.acc_per_channel),
            # "precision_pc": np.array(self.prec_per_channel),
            # "recall_pc": np.array(self.recall_per_channel),
            # "f1score_pc": np.array(self.f1_per_channel),
            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),
            # "hd95_pc": np.array(self.hd95_per_channel),
            # "accuracy": np.array(self.acc_reduced),
            # "precision": np.array(self.prec_reduced),
            # "recall": np.array(self.recall_reduced),
            # "f1score": np.array(self.f1_reduced),
            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.array(self.msd_reduced),
            "ravd": np.array(self.ravd_reduced),
            # "hd95": np.array(self.hd95_reduced),
            "fname": self.fnames,
        }
        return metrics


import numpy as np
import torch
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity, BinaryConfusionMatrix
)


class ClassificationMetrics:
    def __init__(self):
        self.fnames = []
        self.keys = []
        self.preds = []
        self.gts = []

    def update_metrics(self, pred, gt, fname, key):
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()

        if pred.ndim > 1 and pred.shape[1] == 2:
            pred = torch.argmax(pred, dim=1)

        self.fnames.append(fname)
        self.keys.append(key)
        self.preds.append(pred)
        self.gts.append(gt)

    def get_metrics(self):
        preds = torch.cat(self.preds, dim=0)
        gts = torch.cat(self.gts, dim=0)

        # Overall metrics
        accuracy_metric = BinaryAccuracy()
        precision_metric = BinaryPrecision()
        recall_metric = BinaryRecall()
        f1_metric = BinaryF1Score()
        specificity_metric = BinarySpecificity()
        confusion_matrix_metric = BinaryConfusionMatrix()

        overall_accuracy = accuracy_metric(preds, gts).item()
        overall_precision = precision_metric(preds, gts).item()
        overall_recall = recall_metric(preds, gts).item()
        overall_f1 = f1_metric(preds, gts).item()
        overall_specificity = specificity_metric(preds, gts).item()
        overall_balanced_accuracy = (overall_recall + overall_specificity) / 2

        overall_cm = confusion_matrix_metric(preds, gts)  # shape [2,2]
        tn, fp, fn, tp = overall_cm.flatten().numpy()
        if (fp * fn) > 0:
            overall_dor = (tp * tn) / (fp * fn)
        else:
            overall_dor = float('nan')

        # Per-joint
        joint_preds = {}
        joint_gts = {}

        for fname, joint, pred, gt in zip(self.fnames, self.keys, self.preds, self.gts):
            if joint not in joint_preds:
                joint_preds[joint] = []
                joint_gts[joint] = []

            joint_preds[joint].append(pred)
            joint_gts[joint].append(gt)

        joint_metrics = {}
        for joint in joint_preds.keys():
            joint_pred = torch.cat(joint_preds[joint], dim=0)
            joint_gt = torch.cat(joint_gts[joint], dim=0)

            acc = accuracy_metric(joint_pred, joint_gt).item()
            prec = precision_metric(joint_pred, joint_gt).item()
            rec = recall_metric(joint_pred, joint_gt).item()
            f1 = f1_metric(joint_pred, joint_gt).item()
            specificity = specificity_metric(joint_pred, joint_gt).item()
            balanced_accuracy = (rec + specificity) / 2

            cm = confusion_matrix_metric(joint_pred, joint_gt)
            tn, fp, fn, tp = cm.flatten().tolist()
            if (fp * fn) > 0:
                dor = (tp * tn) / (fp * fn)
            else:
                dor = float('nan')

            joint_metrics[joint] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1score": f1,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "dor": dor,
                "confusion_matrix": [tn, fp, fn, tp]  # ⭐⭐ 直接保存
            }

        return {
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_specificity": overall_specificity,
            "overall_balanced_accuracy": overall_balanced_accuracy,
            "overall_dor": overall_dor,
            "overall_confusion_matrix": overall_cm.flatten().tolist(),
            "joint_metrics": joint_metrics,
            "fname": self.fnames,
            "joint": self.keys,
        }



if __name__ == '__main__':
    import torch
    ravd = RAVDMetric(True)
    gt = np.zeros((1, 1,100, 100), dtype=np.uint8)
    gt[..., 20:80, 20:80] = 1  # 真值区域：60x60
    gt = torch.tensor(gt)
    pred = np.zeros((1, 1,100, 100), dtype=np.uint8)
    pred[..., 25:75, 25:75] = 1  # 预测区域：50x50
    pred = torch.tensor(pred)
    value = ravd(pred, gt)
    print("RAVD =", value)
