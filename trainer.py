import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import show_mask
import monai
import pandas as pd
from evaluations.metrics import SegmentationMetrics, ClassificationMetrics, bone_name_dict
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import time
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


class SegTrainer:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0"):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.device = device
        self.max_epoch = args.max_epoch
        self.dice_metric = monai.metrics.DiceMetric(reduction="none")
        self.scaler = GradScaler() if self.amp else None
        self.earlystop = EarlyStopping(patience=10)
        if args.scheduler == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01)
        elif args.scheduler == "Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, cooldown=2)
        else:
            self.scheduler = None

    def fit(self, args):
        train_loss = []
        val_loss = []
        best_val_loss = np.Inf
        for epoch in range(self.max_epoch):
            if next(self.net.parameters()).device != self.device:
                self.net = self.net.to(self.device)
            if self.amp:
                epoch_train_loss_reduced = self.train_one_epoch_amp(epoch)
            else:
                epoch_train_loss_reduced = self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss_reduced)
            epoch_val_loss_reduced = self.validate(epoch)
            if self.earlystop(epoch_val_loss_reduced):
                break
            val_loss.append(epoch_val_loss_reduced)
            self.plot(args, train_loss, val_loss)
            if args.scheduler == "CosineAnnealing":
                self.scheduler.step()
            elif args.scheduler == "Plateau":
                self.scheduler.step(epoch_val_loss_reduced)
            ckpt = {
                "model": self.net.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "train_loss": epoch_train_loss_reduced,
                "val_loss": epoch_val_loss_reduced,
            }
            if epoch_val_loss_reduced < best_val_loss:
                torch.save(ckpt, (Path(args.model_save_path) / "model_best.pth"))
                print(f"New best val loss: {best_val_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
                best_val_loss = epoch_val_loss_reduced
            else:
                torch.save(ckpt, (Path(args.model_save_path) / "model_latest.pth"))
                print(f"Best val_loss didn't decrease, current val_loss: {epoch_val_loss_reduced:.4f}, best val_loss: {best_val_loss:.4f}")

    def train_one_epoch_amp(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            # Avoid non-binary value caused by resize
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            with autocast():
                pred = self.net(img)
                loss = self.criterion(pred, gt)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                 f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def train_one_epoch(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            # Avoid non-binary value caused by resize
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            pred = self.net(img)
            loss = self.criterion(pred, gt)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                 f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.net.eval()
        dice_scores = []
        pbar = tqdm(self.val_loader)
        avg_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred = self.net(img)
                loss = self.criterion(pred, gt)
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                dice_score_single = self.dice_metric(pred_bin, gt).squeeze().cpu().numpy().mean()
                dice_scores.append(dice_score_single)
                avg_loss += loss.item()
                pbar.set_description(f"Epoch {epoch} Validating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        dice_score_reduced = np.array(dice_scores).mean()
        print("Dice: ", dice_score_reduced)
        return avg_loss

    def plot(self, args, train_loss, val_loss):
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class SegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        # self.bone_dsc = []
        # self.dsc_reduced = []
        # self.bone_nsd = []
        # self.nsd_reduced = []
        # self.bone_hd95 = []
        # self.hd95_reduced = []
        # self.bone_acc = []
        # self.acc_reduced = []
        # self.bone_prec = []
        # self.prec_reduced = []
        # self.bone_recall = []
        # self.recall_reduced = []
        # self.bone_f1 = []
        # self.f1_reduced = []
        # self.DSC = monai.metrics.DiceMetric(reduction="none")
        # self.NSD = monai.metrics.SurfaceDistanceMetric(include_background=True, reduction="none")
        # self.HD95 = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
        self.metrics = SegmentationMetrics(num_classes=14)
        # self.Precision = monai.metrics.
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        total_infer_time = 0
        total_items = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                start_time = time.time()  # ⏱️ Start timing
                pred = self.net(img)
                end_time = time.time()  # ⏱️ End timing
                infer_time = end_time - start_time
                total_infer_time += infer_time
                total_items += img.shape[0]  # 批量大小
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                self.metrics.update_metrics(pred_bin, gt, batch["fname"][0])
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.save_overlay:
                    self.create_overlay(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
                if self.save_pred:
                    self.create_pred(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
                    self.create_overlay_single(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
        if self.save_csv:
            self.create_csv(self.args)
        metrics_dict = self.metrics.get_metrics()
        dsc_reduced = metrics_dict["dsc"].mean()
        print("Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict["nsd"].mean()
        print("Mean NSD: ", nsd_reduced)

        avg_infer_time = total_infer_time / total_items
        print(f"Average inference time per item: {avg_infer_time * 1000:.2f} ms")

    def create_overlay(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            # color = np.random.rand(3)
            # seg = torch.sigmoid(pred_mask[0][i]).cpu().numpy()
            # seg[seg > 0.5] = 1
            # seg[seg <= 0.5] = 0
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(self.colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '.pdf'), dpi=600)
        plt.close()

    def create_pred(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "pred"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_overlay_single(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay_single"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])


        # acc_df = pd.DataFrame(metrics_dict["accuracy_pc"], columns=[f"Accuracy {bone_name_dict[i]}" for i in range(num_classes)])
        # acc_mean_df = pd.DataFrame(metrics_dict["accuracy"], columns=["Mean Accuracy"])
        # precision_df = pd.DataFrame(metrics_dict["precision_pc"], columns=[f"Precision {bone_name_dict[i]}" for i in range(num_classes)])
        # precision_mean_df = pd.DataFrame(metrics_dict["precision"], columns=["Mean Precision"])
        # recall_df = pd.DataFrame(metrics_dict["recall_pc"], columns=[f"Recall {bone_name_dict[i]}" for i in range(num_classes)])
        # recall_mean_df = pd.DataFrame(metrics_dict["recall"], columns=["Mean Recall"])
        # f1_df = pd.DataFrame(metrics_dict["f1score_pc"], columns=[f"F1-score {bone_name_dict[i]}" for i in range(num_classes)])
        # f1_mean_df = pd.DataFrame(metrics_dict["f1score"], columns=["Mean F1-score"])
        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
         [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df,
               ravd_df, ravd_mean_df], axis=1)
        # metric_df = pd.concat(
        #     [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, hd95_df, hd95_mean_df, acc_df, acc_mean_df,
        #      precision_df, precision_mean_df, recall_df, recall_mean_df, f1_df,f1_mean_df], axis=1)
        column_means = metric_df.iloc[:, 1:].mean()
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class SegInferer:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_csv = args.save_csv
        self.metrics = SegmentationMetrics(num_classes=14)
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred = self.net(img)
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                self.metrics.update_metrics(pred_bin, gt, batch["fname"][0])
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.save_overlay:
                    self.create_overlay(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
        if self.save_csv:
            self.create_csv(self.args)
        metrics_dict = self.metrics.get_metrics()
        dsc_reduced = metrics_dict["dsc"].mean()
        print("Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict["nsd"].mean()
        print("Mean NSD: ", nsd_reduced)

    def create_overlay(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)
        colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '.pdf'), dpi=600)
        plt.close()



    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])


        # acc_df = pd.DataFrame(metrics_dict["accuracy_pc"], columns=[f"Accuracy {bone_name_dict[i]}" for i in range(num_classes)])
        # acc_mean_df = pd.DataFrame(metrics_dict["accuracy"], columns=["Mean Accuracy"])
        # precision_df = pd.DataFrame(metrics_dict["precision_pc"], columns=[f"Precision {bone_name_dict[i]}" for i in range(num_classes)])
        # precision_mean_df = pd.DataFrame(metrics_dict["precision"], columns=["Mean Precision"])
        # recall_df = pd.DataFrame(metrics_dict["recall_pc"], columns=[f"Recall {bone_name_dict[i]}" for i in range(num_classes)])
        # recall_mean_df = pd.DataFrame(metrics_dict["recall"], columns=["Mean Recall"])
        # f1_df = pd.DataFrame(metrics_dict["f1score_pc"], columns=[f"F1-score {bone_name_dict[i]}" for i in range(num_classes)])
        # f1_mean_df = pd.DataFrame(metrics_dict["f1score"], columns=["Mean F1-score"])
        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
         [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df,
               ravd_df, ravd_mean_df], axis=1)
        # metric_df = pd.concat(
        #     [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, hd95_df, hd95_mean_df, acc_df, acc_mean_df,
        #      precision_df, precision_mean_df, recall_df, recall_mean_df, f1_df,f1_mean_df], axis=1)
        column_means = metric_df.iloc[:, 1:].mean()
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class RegTrainer:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0"):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.device = device
        self.max_epoch = args.max_epoch
        self.scaler = GradScaler() if self.amp else None
        self.earlystop = EarlyStopping(patience=10)
        if args.scheduler == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01)
        elif args.scheduler == "Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, cooldown=2)
        else:
            self.scheduler = None

    def fit(self, args):
        train_loss = []
        val_loss = []
        best_val_loss = np.Inf
        for epoch in range(self.max_epoch):
            if next(self.net.parameters()).device != self.device:
                self.net = self.net.to(self.device)
            if self.amp:
                epoch_train_loss_reduced = self.train_one_epoch_amp(epoch)
            else:
                epoch_train_loss_reduced = self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss_reduced)
            epoch_val_loss_reduced = self.validate(epoch)
            if self.earlystop(epoch_val_loss_reduced):
                break
            val_loss.append(epoch_val_loss_reduced)
            self.plot(args, train_loss, val_loss)
            if args.scheduler == "CosineAnnealing":
                self.scheduler.step()
            elif args.scheduler == "Plateau":
                self.scheduler.step(epoch_val_loss_reduced)
            ckpt = {
                "model": self.net.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "train_loss": epoch_train_loss_reduced,
                "val_loss": epoch_val_loss_reduced,
            }
            if epoch_val_loss_reduced < best_val_loss:
                torch.save(ckpt, (Path(args.model_save_path) / "model_best.pth"))
                print(f"New best val loss: {best_val_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
                best_val_loss = epoch_val_loss_reduced
            else:
                torch.save(ckpt, (Path(args.model_save_path) / "model_latest.pth"))
                print(f"Best val_loss didn't decrease, current val_loss: {epoch_val_loss_reduced:.4f}, best val_loss: {best_val_loss:.4f}")

    def train_one_epoch_amp(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            with autocast():
                # pred = torch.sigmoid(self.net(img))*5.0
                pred = self.net(img)
                if type(pred) != torch.Tensor:
                    pred = pred[0]
                # loss = self.criterion(torch.relu(pred), gt)
                loss = self.criterion(pred, gt)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                 f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def train_one_epoch(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            pred = self.net(img)
            loss = self.criterion(pred, gt)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                 f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.net.eval()
        pbar = tqdm(self.val_loader)
        avg_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred = self.net(img)
                if type(pred) != torch.Tensor:
                    pred = pred[0]
                # pred = torch.sigmoid(self.net(img)) * 5.0
                # loss = self.criterion(torch.relu(pred), gt)
                loss = self.criterion(pred, gt)
                avg_loss += loss.item()
                pbar.set_description(f"Epoch {epoch} Validating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def plot(self, args, train_loss, val_loss):
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class RegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=args.num_classes).to(device)
        self.save_csv = args.save_csv
        self.metrics = ClassificationMetrics()
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        total_infer_time = 0
        total_items = 0
        all_entries = []  # �� 收集所有 joint 的记录

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]  # [B] or [B,1]
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)

                start_time = time.time()
                pred = self.net(img)
                end_time = time.time()
                infer_time = end_time - start_time
                total_infer_time += infer_time
                total_items += img.shape[0]

                if type(pred) != torch.Tensor:
                    pred = pred[0]
                pred_class = torch.argmax(pred, dim=1)

                self.confusion_matrix.update(pred_class, gt)
                self.metrics.update_metrics(pred, gt, batch["fname"][0], batch["key"][0])

                # �� 收集所有 joint 的 GT 和预测
                for i in range(gt.shape[0]):
                    all_entries.append({
                        "Filename": Path(batch["fname"][i]).parent,
                        "Joint": batch["key"][i],
                        "GT": int(gt[i].item()),
                        "Pred": int(pred_class[i].item())
                    })

                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # �� 保存匹配条件的文件夹的全部 joint 信息
        if self.save_csv:
            output_path = Path(self.args.checkpoint) / "Matched_Folder_AllJoints.csv"
            self.collect_matched_folders(all_entries, output_path)

        confmat = self.confusion_matrix.compute().cpu().numpy()
        print("Confusion Matrix:\n", confmat)
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=["w/o BE", "w/ BE"])
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(Path(self.args.checkpoint) / "ConfusionMatrix.pdf")

        avg_infer_time = total_infer_time / total_items
        print(f"Average inference time per item: {avg_infer_time * 1000:.2f} ms")

        if self.save_csv:
            self.create_csv(self.args)
            self.plot_and_save_confusions(self.args)

    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()

        # 总体指标
        overall_accuracy = metrics_dict["overall_accuracy"]
        overall_precision = metrics_dict["overall_precision"]
        overall_recall = metrics_dict["overall_recall"]
        overall_f1 = metrics_dict["overall_f1"]
        overall_specificity = metrics_dict["overall_specificity"]
        overall_balanced_accuracy = metrics_dict["overall_balanced_accuracy"]
        overall_dor = metrics_dict["overall_dor"]

        joint_metrics = metrics_dict["joint_metrics"]

        joint_rows = []
        for joint, m in joint_metrics.items():
            row = {
                "Joint": joint,
                "Mean Accuracy": m["accuracy"],
                "Mean Precision": m["precision"],
                "Mean Recall": m["recall"],
                "Mean F1score": m["f1score"],
                "Mean Specificity": m["specificity"],
                "Mean Balanced Accuracy": m["balanced_accuracy"],
                "Mean DOR": m["dor"],
            }
            joint_rows.append(row)

        joint_df = pd.DataFrame(joint_rows)

        overall_row = pd.DataFrame([{
            "Joint": "Average",
            "Mean Accuracy": overall_accuracy,
            "Mean Precision": overall_precision,
            "Mean Recall": overall_recall,
            "Mean F1score": overall_f1,
            "Mean Specificity": overall_specificity,
            "Mean Balanced Accuracy": overall_balanced_accuracy,
            "Mean DOR": overall_dor,
        }])

        final_df = pd.concat([joint_df, overall_row], ignore_index=True)

        final_df.to_csv(save_path / 'test_metrics.csv', index=False)

    def plot_and_save_confusions(self, args):
        save_path = Path(args.checkpoint)
        save_path.mkdir(exist_ok=True, parents=True)

        metrics_dict = self.metrics.get_metrics()
        joint_metrics = metrics_dict["joint_metrics"]

        for joint, m in joint_metrics.items():
            tn, fp, fn, tp = m["confusion_matrix"]

            confmat = np.array([[tn, fp],
                                [fn, tp]])

            disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=["w/o BE", "w/ BE"])
            fig, ax = plt.subplots(figsize=(4, 4))
            disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
            ax.set_title(f"Confusion Matrix: {joint}")
            plt.tight_layout()

            # 保存PDF
            fig.savefig(save_path / f'{joint}_confusion.pdf')
            plt.close(fig)

    def collect_matched_folders(self, all_entries, output_path):
        """
        找出至少一个 joint 满足 gt==1 且 pred==1 的文件夹（fname），
        然后将这些文件夹下所有 joint 的预测结果保存为 CSV。
        """
        from collections import defaultdict
        import pandas as pd

        # 1. 收集所有 entries 按 fname 分组
        fname_to_entries = defaultdict(list)
        positive_fnames = set()

        for entry in all_entries:
            fname = entry["Filename"]
            fname_to_entries[fname].append(entry)

            # 如果有任何一个 joint 满足 gt==1 且 pred==1，记录下来
            if entry["GT"] == 1 and entry["Pred"] == 1:
                positive_fnames.add(fname)

        # 2. 筛选所有满足条件的 fname 的全部 joint 条目
        matched_entries = []
        for fname in positive_fnames:
            matched_entries.extend(fname_to_entries[fname])

        # 3. 写入 CSV
        if matched_entries:
            matched_df = pd.DataFrame(matched_entries)
            matched_df.to_csv(output_path, index=False)
            print(f"Saved matched folder entries to {output_path}")
        else:
            print("No matched folders found.")


class EarlyStopping:
    def __init__(self, patience, delta=0.0, mode="min"):
        self.patience = patience
        self.mode = mode
        self.delta = 0.0
        self.best_score = None
        self.counter = 0

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
