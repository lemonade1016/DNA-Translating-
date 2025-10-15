import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import random
import pytorch_lightning as pl

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 6
BATCH_SIZE = 64 
NUM_SAMPLES_PER_EPOCH = 2000
NUM_EPOCHS = 500
PATIENCE = 50
SEQ_LENGTH = 59
INPUT_DIM = 2048 
LEARNING_RATE = 0.001
L2_REG_WEIGHT = 1e-4
GRAD_CLIP_NORM = 1.0


TRAIN_DATA_ROOT = ''
PREDICTOR_MODEL_PATH = ''
MODEL_SAVE_PATH = ''

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DNAEncoder(nn.Module):
    def __init__(self, input_dim, seq_length=59, num_nucleotides=4):
        super(DNAEncoder, self).__init__()
        self.seq_length = seq_length
        self.num_nucleotides = num_nucleotides
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, seq_length * num_nucleotides)
        )
        self.layer_norm = nn.LayerNorm(seq_length * num_nucleotides)

    def forward(self, x, temperature=1.5):
        logits = self.fc(x)
        logits = self.layer_norm(logits)
        logits = logits.view(-1, self.seq_length, self.num_nucleotides)
        soft_seq = F.softmax(logits, dim=-1)
        gumbel_output = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return gumbel_output.permute(0, 2, 1), soft_seq.permute(0, 2, 1)

class ImageToDNA(nn.Module):
    def __init__(self, num_classes, input_dim, seq_length):
        super(ImageToDNA, self).__init__()
        resnet_model = torchvision.models.resnet50(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])
        self.dna_encoder = DNAEncoder(input_dim=input_dim, seq_length=seq_length)
    
    def forward(self, x, temperature=1.5):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        
        dna_seq, soft_seq = self.dna_encoder(features, temperature)
        return dna_seq, soft_seq

class CNNRegression(pl.LightningModule):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(2, 1280, kernel_size=(4, 9), padding=(0, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(1280),
            nn.Dropout2d(0.4),
        )
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(1280, 512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(10),
        )
        self.residual_conv1 = nn.Conv1d(1280, 128, kernel_size=1)
        self.residual_conv2 = nn.Conv1d(128, 64, kernel_size=1)
        self.lin_block = nn.Sequential(
            nn.Linear(64 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv2d_block(x)
        x = x.squeeze(2)
        residual1 = x
        x = self.conv1d_block[0:3](x)
        x = self.conv1d_block[3:9](x)
        residual1 = self.residual_conv1(residual1)
        x = x + residual1
        residual2 = x
        x = self.conv1d_block[9:16](x)
        residual2 = self.residual_conv2(residual2)
        x = x + residual2
        x = self.conv1d_block[16:](x)
        x = x.view(x.size(0), -1)
        x = self.lin_block(x)
        return x

def differentiable_consecutive_penalty(soft_seq, window=3, threshold=0.3):
    B, _, L = soft_seq.shape
    total_penalty = 0.0
    loop_range = L - window + 1
    if loop_range <= 0:
        return torch.tensor(0.0, device=soft_seq.device)

    for i in range(loop_range):
        window_prob = soft_seq[:, :, i:i + window]
        prod_prob = torch.prod(window_prob, dim=2)
        max_prod, _ = prod_prob.max(dim=1)
        penalty = torch.relu(max_prod - threshold).mean()
        total_penalty += penalty
    
    if loop_range == 0:
        return torch.tensor(0.0, device=soft_seq.device)
        
    return total_penalty / loop_range

def compute_loss(dna_seq1, soft_seq1, dna_seq2, soft_seq2, labels, predictor_model, consecutive_weight, diff_class_weight, same_class_weight):
    seq_pairs = torch.stack([dna_seq1, dna_seq2], dim=1)
    predicted_yields = predictor_model(seq_pairs).squeeze()
    
    same_class_mask = (labels == 1).float()
    diff_class_mask = (labels == 0).float()

    same_class_loss = ((1 - predicted_yields) * same_class_mask).pow(2).sum() / (same_class_mask.sum() + 1e-8)
    diff_class_loss = (predicted_yields * diff_class_mask).pow(2).sum() / (diff_class_mask.sum() + 1e-8)

    consecutive_penalty = 0.0
    if consecutive_weight > 0:
        for window, factor in zip([3, 4, 5], [1.0, 10.0, 100.0]):
            penalty1 = differentiable_consecutive_penalty(soft_seq1, window=window, threshold=0.3)
            penalty2 = differentiable_consecutive_penalty(soft_seq2, window=window, threshold=0.3)
            consecutive_penalty += (penalty1 + penalty2) * factor
        consecutive_penalty /= 2

    total_loss = same_class_weight * same_class_loss + diff_class_weight * diff_class_loss + consecutive_weight * consecutive_penalty
    return total_loss, same_class_loss.item(), diff_class_loss.item(), consecutive_penalty.item()

class ImagePairDataset(Dataset):
    def __init__(self, image_folder_dataset):
        self.image_folder_dataset = image_folder_dataset
        self.labels = [s[1] for s in image_folder_dataset.samples]
        self.indices = list(range(len(image_folder_dataset)))

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, index):
        return self.image_folder_dataset[index]

class BalancedPairSampler(Sampler):
    def __init__(self, dataset, num_samples_per_epoch):
        self.dataset = dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.labels = np.array(dataset.labels)
        self.indices_by_class = [np.where(self.labels == i)[0] for i in range(NUM_CLASSES)]

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples_per_epoch):
            if random.random() < 0.5: 
                class_idx = random.randrange(NUM_CLASSES)
                idx1, idx2 = random.sample(list(self.indices_by_class[class_idx]), 2)
                indices.append((idx1, idx2, 1))
            else: 
                class1_idx, class2_idx = random.sample(range(NUM_CLASSES), 2)
                idx1 = random.choice(self.indices_by_class[class1_idx])
                idx2 = random.choice(self.indices_by_class[class2_idx])
                indices.append((idx1, idx2, 0))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_epoch

def collate_fn(batch):
    indices_pairs = batch
    img1_list, img2_list, label_list = [], [], []
    for idx1, idx2, label in indices_pairs:
        img1, _ = train_dataset[idx1]
        img2, _ = train_dataset[idx2]
        img1_list.append(img1)
        img2_list.append(img2)
        label_list.append(label)
    
    return torch.stack(img1_list), torch.stack(img2_list), torch.tensor(label_list, dtype=torch.long)

if __name__ == '__main__':
    train_image_folder = datasets.ImageFolder(root=TRAIN_DATA_ROOT, transform=data_transforms)
    train_dataset = ImagePairDataset(train_image_folder)
    train_sampler = BalancedPairSampler(train_dataset, num_samples_per_epoch=NUM_SAMPLES_PER_EPOCH)
    train_loader = DataLoader(train_sampler, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

    model = ImageToDNA(num_classes=NUM_CLASSES, input_dim=INPUT_DIM, seq_length=SEQ_LENGTH).to(DEVICE)
    
    predictor_model = CNNRegression()
    predictor_model.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE, weights_only=True))
    predictor_model.to(DEVICE).eval()
    for param in predictor_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.dna_encoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        model.feature_extractor.eval() 

        total_loss, total_hyb_loss, total_diff_loss, total_cons_loss = 0, 0, 0, 0
        temperature = max(1.5 - epoch * 0.002, 0.5)

        for i, (images1, images2, labels) in enumerate(train_loader):
            images1, images2, labels = images1.to(DEVICE), images2.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            dna_seq1, soft_seq1 = model(images1, temperature=temperature)
            dna_seq2, soft_seq2 = model(images2, temperature=temperature)
            
            loss, hyb_loss, diff_loss, cons_loss = compute_loss(
                dna_seq1, soft_seq1, dna_seq2, soft_seq2, labels, predictor_model,
                CONSECUTIVE_WEIGHT, DIFF_CLASS_WEIGHT, SAME_CLASS_WEIGHT
            )
            
            l2_reg = L2_REG_WEIGHT * sum(p.norm(2) for p in model.dna_encoder.parameters())
            loss += l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dna_encoder.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_hyb_loss += hyb_loss
            total_diff_loss += diff_loss
            total_cons_loss += cons_loss

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_hyb_loss = total_hyb_loss / num_batches
        avg_diff_loss = total_diff_loss / num_batches
        avg_cons_loss = total_cons_loss / num_batches

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Same-Class Loss: {avg_hyb_loss:.4f}, "
              f"Diff-Class Loss: {avg_diff_loss:.4f}, Consecutive Loss: {avg_cons_loss:.4f}")

        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
