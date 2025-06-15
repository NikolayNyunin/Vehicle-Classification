import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torch.optim import SGD
from torchvision.ops import box_iou
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

#Dataset для Stanford Cars
class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, labels_df, transforms=None):
        self.root = root_dir
        self.df = labels_df.reset_index(drop=True)
        self.transforms = transforms
        self.class2idx = {c: i+1 for i, c in enumerate(self.df['general_class_name'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, 'images', row['file_name'])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        boxes = torch.tensor([[row['boxx_1'], row['boxy_1'], row['boxx_2'], row['boxy_2']]], dtype=torch.float32)
        labels = torch.tensor([self.class2idx[row['general_class_name']]], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img, target

#collate_fn для DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

#Построение графика loss по эпохам
def plot_loss_curve(epoch_losses):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

#Тренировка Faster R-CNN с прогресс-баром по batch
def train(model, train_loader, optimizer, device, num_epochs):
    model.to(device)
    epoch_losses = []
    checkpoint_path = 'faster_rcnn_final.pt'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (images, targets) in enumerate(progress, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / batch_idx)

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} finished, avg loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    plot_loss_curve(epoch_losses)
    return epoch_losses

#Вычисление метрики F1
def evaluate_f1(model, data_loader, device,
                iou_thresh=0.5,
                score_thresh=0.05):
    model.eval()
    TP = FP = FN = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Eval", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                keep = out["scores"] > score_thresh
                pred_boxes = out["boxes"][keep].cpu()
                pred_labels = out["labels"][keep].cpu()

                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()

                if len(pred_boxes) == 0:
                    FN += len(gt_boxes)
                    continue

                iou_mat = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                for p_idx, p_cls in enumerate(pred_labels):
                    mask = (gt_labels == p_cls)
                    if matched_gt:
                        mask[list(matched_gt)] = False
                    if mask.sum() == 0:
                        FP += 1
                        continue

                    ious = iou_mat[p_idx, mask]
                    if ious.max() >= iou_thresh:
                        TP += 1
                        global_idx = torch.where(mask)[0][ious.argmax()].item()
                        matched_gt.add(global_idx)
                    else:
                        FP += 1

                FN += (len(gt_boxes) - len(matched_gt))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1
