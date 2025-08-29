from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
import monai
import numpy as np
import torch
from itertools import combinations
from monai.metrics.metric import CumulativeIterationMetric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity, BinaryConfusionMatrix

bone_name_dict = {0: "Capitate", 1: "DistalRadius", 2: "DistalUlna", 3: "Hamate", 4: "Lunate", 5: "Pisifrom&Triquetrum",
                  6: "Scaphoid", 7: "Trapzium", 8: "Trapzoid", 9: "metacarpal1st", 10: "metacarpal2nd",
                  11: "metacarpal3rd", 12: "metacarpal4th", 13: "metacarpal5th"}

overlap_pairs = [(1, 6), (1, 4), (6, 7), (0, 6), (7, 9), (0, 11), (3, 12), (4, 6), (7, 8), (0, 8),
                 (3, 13), (7, 10), (8, 10), (10, 11)]  # , (1, 2), (6, 8), (9, 10)


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


class OverlapMetric:
    def __init__(self, nsd_tolerance, num_classes):
        self.nsd_tolerance = nsd_tolerance
        self.num_classes = num_classes

        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[self.nsd_tolerance],
            include_background=True,
            reduction="none"
        )

    def __call__(self, pred_bin, gt_bin):
        dsc_per_pair = []
        nsd_per_pair = []
        avg_dsc = []
        avg_nsd = []
        valid_pairs = []

        for b in range(gt_bin.shape[0]):
            dsc_per_pair_ = []
            nsd_per_pair_ = []
            valid_pairs_ = []

            pred_overlap_list = []
            gt_overlap_list = []

            for i, j in combinations(range(self.num_classes), 2):
                gt_i = gt_bin[b][i] > 0
                gt_j = gt_bin[b][j] > 0
                gt_overlap = torch.logical_and(gt_i, gt_j)

                if not gt_overlap.any():
                    continue  # skip non-overlapping pairs in GT

                pred_i = pred_bin[b][i] > 0
                pred_j = pred_bin[b][j] > 0
                pred_overlap = torch.logical_and(pred_i, pred_j)

                # 构造 [1, 1, H, W] 格式以兼容 MONAI Metric
                gt_tensor = gt_overlap[None, None].float()
                pred_tensor = pred_overlap[None, None].float()

                gt_overlap_list.append(gt_tensor)
                pred_overlap_list.append(pred_tensor)
                valid_pairs_.append((i, j))

            if len(gt_overlap_list) == 0:
                # 当前样本没有任何 valid pair
                avg_dsc.append(0.0)
                avg_nsd.append(0.0)
                dsc_per_pair.append([])
                nsd_per_pair.append([])
                valid_pairs.append([])
                continue

            # 堆叠成 [N, 1, H, W]
            pred_stack = torch.cat(pred_overlap_list, dim=0)
            gt_stack = torch.cat(gt_overlap_list, dim=0)

            # 计算 Dice
            dsc_values = self.dsc(pred_stack, gt_stack).detach().cpu()

            # 计算 NSD
            nsd_values = self.nsd(pred_stack, gt_stack).detach().cpu()

            # 处理 NSD 中的 nan 值（替换为 0.0）
            nsd_values = torch.nan_to_num(nsd_values, nan=0.0)

            for i in range(len(valid_pairs_)):
                dsc_per_pair_.append(dsc_values[i].item())
                nsd_per_pair_.append(nsd_values[i].item())

            avg_dsc_ = dsc_values.mean().item()
            avg_nsd_ = nsd_values.mean().item()

            dsc_per_pair.append(dsc_per_pair_)
            nsd_per_pair.append(nsd_per_pair_)
            avg_dsc.append(avg_dsc_)
            avg_nsd.append(avg_nsd_)
            valid_pairs.append(valid_pairs_)

        return avg_dsc, avg_nsd, dsc_per_pair, nsd_per_pair, valid_pairs

    def get_valid_pairs(self, gt_bin):
        valid_pairs = []
        for i, j in combinations(range(self.num_classes), 2):
            if torch.logical_and(gt_bin[i] > 0, gt_bin[j] > 0).any():
                valid_pairs.append((i, j))
        return valid_pairs


class NewOverlapMetric:
    def __init__(self, nsd_tolerance, num_classes, overlap_pairs):
        self.nsd_tolerance = nsd_tolerance
        self.num_classes = num_classes

        self.dsc = monai.metrics.DiceMetric(reduction="none")
        self.nsd = monai.metrics.SurfaceDiceMetric(
            class_thresholds=[self.nsd_tolerance for _ in range(len(overlap_pairs))],
            include_background=True,
            reduction="none",
            get_not_nans=True
        )
        self.voe = VOEMetric(monai.metrics.MeanIoU(include_background=True, reduction="none"))
        self.msd = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="none", get_not_nans=True)
        self.ravd = RAVDMetric(include_background=True)
        self.valid_pairs = overlap_pairs

    def __call__(self, pred_bin, gt_bin):
        """
        pred_bin, gt_bin: [B, num_classes, H, W]
        overlap_pairs: list of (i, j) tuples specifying which bone pairs to evaluate
        """
        dsc_per_pair = None
        nsd_per_pair = None
        voe_per_pair = None
        msd_per_pair = None
        ravd_per_pair = None
        avg_dsc = []
        avg_nsd = []
        avg_voe = []
        avg_msd = []
        avg_ravd = []

        new_pred_bin = None
        new_gt_bin = None
        for pair in self.valid_pairs:
            if new_pred_bin is None:
                new_pred_bin = torch.logical_and(pred_bin[:, pair[0], ...], pred_bin[:, pair[1], ...]).unsqueeze(1)
                new_gt_bin = torch.logical_and(gt_bin[:, pair[0], ...], gt_bin[:, pair[1], ...]).unsqueeze(1)
            else:
                pred_overlap_bin = torch.logical_and(pred_bin[:, pair[0], ...], pred_bin[:, pair[1], ...]).unsqueeze(1)
                gt_overlap_bin = torch.logical_and(gt_bin[:, pair[0], ...], gt_bin[:, pair[1], ...]).unsqueeze(1)
                new_pred_bin = torch.cat([new_pred_bin, pred_overlap_bin], dim=1)
                new_gt_bin = torch.cat([new_gt_bin, gt_overlap_bin], dim=1)

        dsc_values = self.dsc(new_pred_bin, new_gt_bin).detach().cpu()
        nsd_values = self.nsd(new_pred_bin, new_gt_bin).detach().cpu()

        voe_values = self.voe(new_pred_bin, new_gt_bin).squeeze()
        msd_values = self.msd(new_pred_bin, new_gt_bin).squeeze()
        # 删除：不再把 inf 改成 nan
        # msd_values = torch.where(torch.isfinite(msd_values), msd_values,
        #                          torch.tensor(float('nan'), device=msd_values.device, dtype=msd_values.dtype))
        ravd_values = self.ravd(new_pred_bin, new_gt_bin).squeeze()

        avg_dsc_ = dsc_values.nanmean().item()
        avg_nsd_ = nsd_values.nanmean().item()

        avg_voe_ = voe_values.nanmean().item()
        # 这里用 isfinite 跳过 inf 与 nan；若没有有限值则返回 nan
        _finite = torch.isfinite(msd_values)
        avg_msd_ = msd_values[_finite].mean().item() if _finite.any() else float('nan')
        avg_ravd_ = ravd_values.nanmean().item()

        if dsc_per_pair is None:
            dsc_per_pair = dsc_values
        else:
            dsc_per_pair = torch.cat([dsc_per_pair, dsc_values], dim=0)

        if nsd_per_pair is None:
            nsd_per_pair = nsd_values
        else:
            nsd_per_pair = torch.cat([nsd_per_pair, nsd_values], dim=0)

        if voe_per_pair is None:
            voe_per_pair = voe_values
        else:
            voe_per_pair = torch.cat([voe_per_pair, voe_values], dim=0)

        if msd_per_pair is None:
            msd_per_pair = msd_values
        else:
            msd_per_pair = torch.cat([msd_per_pair, msd_values], dim=0)

        if ravd_per_pair is None:
            ravd_per_pair = ravd_values
        else:
            ravd_per_pair = torch.cat([ravd_per_pair, ravd_values], dim=0)
        avg_dsc.append(avg_dsc_)
        avg_nsd.append(avg_nsd_)

        avg_voe.append(avg_voe_)
        avg_msd.append(avg_msd_)
        avg_ravd.append(avg_ravd_)

        return (np.array(avg_dsc), np.array(avg_nsd), np.array(avg_voe), np.array(avg_msd), np.array(avg_ravd),
                dsc_per_pair.squeeze().numpy(), nsd_per_pair.squeeze().numpy(), voe_per_pair.squeeze().numpy(),
                msd_per_pair.squeeze().numpy(), ravd_per_pair.squeeze().numpy(), self.valid_pairs)


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

        self.overlap_dsc_reduced = []
        self.overlap_nsd_reduced = []
        self.overlap_voe_reduced = []
        self.overlap_msd_reduced = []
        self.overlap_ravd_reduced = []
        self.overlap_dsc_per_pair = []
        self.overlap_nsd_per_pair = []
        self.overlap_voe_per_pair = []
        self.overlap_msd_per_pair = []
        self.overlap_ravd_per_pair = []
        self.overlap_pairs = []
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
        self.overlap_metric = NewOverlapMetric(nsd_tolerance=2, num_classes=num_classes, overlap_pairs=overlap_pairs)

    def update_metrics(self, pred_bin, gt, fname):
        self.fnames.append(fname)
        pred_bin = pred_bin.detach().cpu()
        gt = gt.detach().cpu()

        dsc_pc = self.dsc(pred_bin, gt).squeeze()
        nsd_pc = self.nsd(pred_bin, gt).squeeze()
        voe_pc = self.voe(pred_bin, gt).squeeze()
        msd_pc = self.msd(pred_bin, gt).squeeze()
        ravd_pc = self.ravd(pred_bin, gt).squeeze()
        (avg_dsc, avg_nsd, avg_voe, avg_msd, avg_ravd, dsc_per_pair, nsd_per_pair, voe_per_pair, msd_per_pair,
         ravd_per_pair, valid_pairs) = self.overlap_metric(pred_bin, gt)

        self.dsc_per_channel.append(dsc_pc)
        self.nsd_per_channel.append(nsd_pc)
        self.voe_per_channel.append(voe_pc)
        self.msd_per_channel.append(msd_pc)
        self.ravd_per_channel.append(ravd_pc)
        self.overlap_dsc_per_pair.append(dsc_per_pair)
        self.overlap_nsd_per_pair.append(nsd_per_pair)
        self.overlap_voe_per_pair.append(voe_per_pair)
        self.overlap_msd_per_pair.append(msd_per_pair)
        self.overlap_ravd_per_pair.append(ravd_per_pair)

        self.dsc_reduced.append(dsc_pc.mean())
        self.nsd_reduced.append(nsd_pc.mean())
        self.voe_reduced.append(voe_pc.mean())
        self.msd_reduced.append(msd_pc[np.isfinite(msd_pc)].mean() if np.isfinite(msd_pc).any() else np.nan)
        self.ravd_reduced.append(ravd_pc.mean())
        self.overlap_dsc_reduced.append(avg_dsc)
        self.overlap_nsd_reduced.append(avg_nsd)
        self.overlap_voe_reduced.append(avg_voe)
        self.overlap_msd_reduced.append(avg_msd)
        self.overlap_ravd_reduced.append(avg_ravd)

        # self.overlap_pairs.append(valid_pairs)
        self.overlap_pairs = valid_pairs

    def get_metrics(self):
        metrics = {

            "dsc_pc": np.array(self.dsc_per_channel),
            "nsd_pc": np.array(self.nsd_per_channel),
            "voe_pc": np.array(self.voe_per_channel),
            "msd_pc": np.array(self.msd_per_channel),
            "ravd_pc": np.array(self.ravd_per_channel),
            "overlap_dsc_per_pair": self.overlap_dsc_per_pair,
            "overlap_nsd_per_pair": self.overlap_nsd_per_pair,
            "overlap_voe_per_pair": self.overlap_voe_per_pair,
            "overlap_msd_per_pair": self.overlap_msd_per_pair,
            "overlap_ravd_per_pair": self.overlap_ravd_per_pair,

            "dsc": np.array(self.dsc_reduced),
            "nsd": np.array(self.nsd_reduced),
            "voe": np.array(self.voe_reduced),
            "msd": np.where(np.isfinite(self.msd_reduced), self.msd_reduced, np.nan),
            "ravd": np.array(self.ravd_reduced),
            "overlap_dsc": np.array(self.overlap_dsc_reduced),
            "overlap_nsd": np.array(self.overlap_nsd_reduced),
            "overlap_voe": np.array(self.overlap_voe_reduced),
            "overlap_msd": np.array(self.overlap_msd_reduced),
            "overlap_ravd": np.array(self.overlap_ravd_reduced),
            "overlap_pairs": self.overlap_pairs,
            # "hd95": np.array(self.hd95_reduced),
            "fname": self.fnames,
        }
        return metrics


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
