# Expression Meme Detector

This project combines **Facial Expression Recognition (FER)** with hand gesture detection to trigger pre-selected meme images. It classifies facial expressions into seven emotions: Surprise, Fear, Disgust, Happiness, Sadness, Anger, and Neutral.

---

## Project Structure

```
├── CustomCNN/
│   └── CUSTOM_CNN.ipynb
├── EfficientNetB0/
│   ├── EfficientNet_B0.ipynb
│   ├── EfficientNet_B0_COMBINED.ipynb
│   └── EfficientNet_B0_COMBINED_HALVED.ipynb
├── MobileNetV2/
│   └── MobileNetV2.ipynb
└── Resnet18/
    ├── RESNET18.ipynb
    ├── RESNET18_NO_MIXUP_CUTMIX.ipynb
    └── RESNET18_MAX_MIXUP_CUTMIX.ipynb
```

---

## Dataset

| Dataset | Role |
|---|---|
| **RAF-DB** | Primary dataset — high-quality, well-labeled, reliable |
| **FER2013+** | Supplementary — used to boost minority classes (Fear, Disgust). Contempt excluded. |

Dataset paths used in notebooks:

```python
TRAIN_DIR = '/content/MERGED_DATASET/train'
TEST_DIR  = '/content/MERGED_DATASET/test'
```

---

## Models

### Custom CNN
- **`CustomCNN/CUSTOM_CNN.ipynb`** — Fully custom CNN architecture trained on RAF-DB with 0.5 MixUp/CutMix probability.

### EfficientNet-B0
- **`EfficientNetB0/EfficientNet_B0.ipynb`** — Pretrained EfficientNet-B0 trained on RAF-DB.
- **`EfficientNetB0/EfficientNet_B0_COMBINED.ipynb`** — Trained on merged RAF-DB + FER2013+ dataset.
- **`EfficientNetB0/EfficientNet_B0_COMBINED_HALVED.ipynb`** — Merged dataset with Happiness & Neutral classes halved to reduce imbalance.

### MobileNetV2
- **`MobileNetV2/MobileNetV2.ipynb`** — Pretrained MobileNetV2 trained on RAF-DB with 0.5 MixUp/CutMix probability.

### ResNet-18
- **`Resnet18/RESNET18.ipynb`** — Trained on RAF-DB with 0.5 MixUp/CutMix probability.
- **`Resnet18/RESNET18_NO_MIXUP_CUTMIX.ipynb`** — Trained on RAF-DB with MixUp/CutMix disabled.
- **`Resnet18/RESNET18_MAX_MIXUP_CUTMIX.ipynb`** — Trained on RAF-DB with MixUp/CutMix probability set to 1.0.

---

## Hyperparameters

```python
# General
SEED          = 42
BATCH_SIZE    = 64
EPOCHS        = 40
WARMUP_EPOCHS = 12
VAL_SPLIT     = 0.15
MAX_PATIENCE  = 5

# Learning Rate & Regularisation
LR            = 0.001
FINETUNE_LR   = 0.0001
WEIGHT_DECAY  = 1e-4

# Augmentation — MixUp / CutMix
USE_MIXUP           = True
USE_CUTMIX          = True
MIXUP_CUTMIX_PROB   = 0.5
LABEL_SMOOTHING     = 0.1
MIXUP_ALPHA         = 0.4
CUTMIX_ALPHA        = 1.0

# Augmentation — Geometric
HFLIP_PROB        = 0.5
ROTATION_DEGREES  = 5
AFFINE_TRANSLATE  = (0.03, 0.03)
AFFINE_SCALE      = (0.97, 1.03)

# Augmentation — Color
COLOR_JITTER = (0.1, 0.1, 0.1, 0.02)

# Augmentation — Disabled
GRAYSCALE_PROB       = 0.0
GAUSSIAN_BLUR_PROB   = 0.0
RANDOM_ERASING_PROB  = 0.0
```

---

## Training Pipeline

```python
# 1. Load Dataset
train_data, val_data, test_data = load_RAFDB()

# 2. Apply Data Augmentation
train_data = apply_transforms(train_data)

# 3. Initialise Model
model = EfficientNet_B0(num_classes=7)

# 4. Define Loss & Optimiser
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters())

# 5. Training Loop
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = mixup_cutmix(images, labels)  # optional
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc = evaluate(model, val_loader)

# 6. Evaluate on Test Set
test_acc = evaluate(model, test_loader)

# 7. Export to ONNX
export_model_to_ONNX(model)
```

---

## Usage

1. Get your `kaggle.json` API token from [kaggle.com/settings](https://www.kaggle.com/settings) under **API > Create New Token**, then upload it to download the dataset.

2. **Set dataset paths** in the notebook you want to use:
   ```python
   TRAIN_DIR = '/content/MERGED_DATASET/train'
   TEST_DIR  = '/content/MERGED_DATASET/test'
   ```

3. **Open** the notebook for your chosen model (see [Models](#-models) above).

4. **Run all cells** to train and evaluate the model.

5. **Export** the trained model to ONNX for deployment:
   ```python
   export_model_to_ONNX(model)
   ```