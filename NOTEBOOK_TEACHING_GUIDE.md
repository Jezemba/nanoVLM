# nanoVLM Training Notebook - Teaching Guide

## Vision-Language Models: A Complete Walkthrough

This guide provides a comprehensive breakdown of the `train_nanoVLM.ipynb` notebook for teaching Vision-Language Model (VLM) concepts. Each section explains what the code does, what source files it draws from, why it matters, and key criteria to consider.

---

## Table of Contents

1. [High-Level VLM Architecture Overview](#1-high-level-vlm-architecture-overview)
2. [Block 1: Environment Setup (Clone Repository)](#block-1-environment-setup-clone-repository)
3. [Block 2: Install Dependencies](#block-2-install-dependencies)
4. [Block 3: Hugging Face Authentication](#block-3-hugging-face-authentication)
5. [Block 4: Model Naming](#block-4-model-naming)
6. [Block 5: Imports and Device Setup](#block-5-imports-and-device-setup)
7. [Block 6: Dataloader Creation](#block-6-dataloader-creation)
8. [Block 7: Training Loop](#block-7-training-loop)
9. [Block 8: Configuration Classes](#block-8-configuration-classes)
10. [Block 9: Run Training](#block-9-run-training)

---

## 1. High-Level VLM Architecture Overview

Before diving into the code, understand the three core components of a Vision-Language Model:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Vision-Language Model (VLM)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │   Vision    │    │    Modality      │    │     Language      │  │
│  │   Encoder   │───▶│    Projector     │───▶│      Model        │  │
│  │   (ViT)     │    │     (MLP)        │    │    (SmolLM2)      │  │
│  └─────────────┘    └──────────────────┘    └───────────────────┘  │
│        │                    │                        │              │
│   Image → Patch        Vision → Language         Embeddings →      │
│   Embeddings           Embedding Space           Text Output       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: A VLM doesn't "see" images the way humans do. It converts images into sequences of embeddings that can be processed alongside text tokens by a language model.

---

## Block 1: Environment Setup (Clone Repository)

### Code
```python
import os

if not os.path.isdir('nanoVLM'):
    !git clone https://github.com/huggingface/nanoVLM.git
%cd nanoVLM/
!ls
```

### What's Happening
- Checks if the nanoVLM repository already exists
- Clones the official Hugging Face nanoVLM repository if needed
- Changes the working directory to the cloned repo
- Lists contents to verify successful clone

### Why This Step is Important
- Ensures reproducibility - everyone starts from the same codebase
- The `%cd` magic command persists the directory change across cells
- Idempotent check (`if not os.path.isdir`) prevents re-cloning

### Teaching Points
- **Colab vs Local**: In Google Colab, the filesystem resets on disconnect. This pattern handles fresh sessions.
- **Git in Jupyter**: The `!` prefix runs shell commands; `%` runs Jupyter magic commands

---

## Block 2: Install Dependencies

### Code
```python
!pip -q install torch
!pip -q install gcsfs
!pip -q install datasets==3.5.0
!pip -q install tqdm
!pip -q install huggingface_hub
```

### What's Happening
- Installs PyTorch (deep learning framework)
- Installs `gcsfs` (Google Cloud Storage filesystem - for accessing cloud datasets)
- Installs `datasets` (Hugging Face's data loading library) - pinned to version 3.5.0
- Installs `tqdm` (progress bars)
- Installs `huggingface_hub` (for model/dataset downloads and uploads)

### Why This Step is Important
- **Version pinning** (`datasets==3.5.0`) ensures reproducibility
- The `-q` flag suppresses verbose output for cleaner notebooks

### Important Criteria to Consider
| Dependency | Purpose | Why This Version? |
|------------|---------|-------------------|
| `torch` | Core DL framework | Latest stable for GPU support |
| `datasets` | Data loading | 3.5.0 has specific streaming features |
| `gcsfs` | Cloud storage | For accessing HF datasets stored on GCS |

### Teaching Points
- **Dependency Hell**: VLM training involves many libraries that must be compatible
- **Colab Pre-installed**: Some packages (like torch) may already exist in Colab, but explicit installation ensures correct versions

---

## Block 3: Hugging Face Authentication

### Code
```python
from huggingface_hub import notebook_login
notebook_login()
```

### What's Happening
- Imports the notebook-specific login function from Hugging Face Hub
- Displays an interactive login widget to enter your HF token
- Stores credentials for uploading models later

### Source Files Referenced
This enables the `push_to_hub()` method in `models/vision_language_model.py:252-280`

### Why This Step is Important
- Required for pushing trained models to the Hub
- Tokens provide authenticated access to private datasets/models
- The token is stored locally and persists across cells

### Important Criteria to Consider
- **Token Permissions**: Need "write" access for pushing models
- **Security**: Never hardcode tokens in notebooks; use `notebook_login()` or environment variables
- **Organization Access**: Ensure your token has access to the target organization

### Teaching Points
- **HF Ecosystem**: Hugging Face Hub serves as the central repository for ML models, similar to GitHub for code
- **Authentication Flow**: OAuth-based token system allows fine-grained permissions

---

## Block 4: Model Naming

### Code
```python
hf_model_name = "YOUR-HF-USERNAME/nanoVLM"
```

### What's Happening
- Defines the repository path where your trained model will be uploaded
- Format: `{username_or_org}/{model_name}`

### Why This Step is Important
- Required for `model.push_to_hub(hf_model_name)` at the end of training
- Creates a unique identifier for your model variant

### Important Criteria to Consider
- **Naming Conventions**: Use descriptive names (e.g., `username/nanoVLM-finetuned-chartqa`)
- **Visibility**: Models are public by default; pass `private=True` for private repos
- **Versioning**: Consider including training details in the name

---

## Block 5: Imports and Device Setup

### Code
```python
# nanoVLM Imports
from data.datasets import VQADataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer

import models.config as config
from models.vision_language_model import VisionLanguageModel

# Libraries
import math
import time
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

%reload_ext autoreload
%autoreload 2
```

### What's Happening

#### nanoVLM Imports:

| Import | Source File | Purpose |
|--------|-------------|---------|
| `VQADataset` | `data/datasets.py:107-151` | Processes image-text pairs for VQA training |
| `VQACollator` | `data/collators.py:59-71` | Batches samples with proper padding |
| `ConstantLengthDataset` | `data/advanced_datasets.py:11-237` | Creates fixed-length sequences for efficient training |
| `get_image_processor` | `data/processors.py:20-25` | Creates image preprocessing pipeline |
| `get_tokenizer` | `data/processors.py:8-18` | Loads and configures the text tokenizer |
| `VisionLanguageModel` | `models/vision_language_model.py:21-311` | The complete VLM architecture |

#### Device Detection:
```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU (fallback)
```

#### Random Seed Setting:
- `torch.manual_seed(0)` - CPU operations reproducibility
- `torch.cuda.manual_seed_all(0)` - GPU operations reproducibility across all devices

### Why This Step is Important

1. **Tokenizer Parallelism Warning**: Setting `TOKENIZERS_PARALLELISM=false` prevents deadlocks when DataLoader uses multiple workers with HF tokenizers

2. **Device Hierarchy**: The priority order ensures optimal hardware utilization:
   - CUDA provides best performance for NVIDIA GPUs
   - MPS enables GPU acceleration on Apple Silicon Macs
   - CPU is the universal fallback

3. **Reproducibility**: Setting random seeds ensures:
   - Same weight initialization
   - Same data shuffling (when combined with dataset shuffle seed)
   - Comparable results across runs

4. **Autoreload**: `%autoreload 2` automatically reloads modules when source files change - essential for iterative development

### Source Files Deep Dive

#### `data/processors.py` - Image and Text Processing

```python
def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    tokenizer = AutoTokenizer.from_pretrained(name, ...)
    tokenizer.pad_token = tokenizer.eos_token  # Critical for batching!
    return tokenizer

def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len):
    return transforms.Compose([
        DynamicResize(...),      # Resize maintaining aspect ratio
        transforms.ToTensor(),    # PIL → Tensor [0,1]
        GlobalAndSplitImages(...) # Create global + local patches
    ])
```

**Key Insight**: The image processor creates BOTH a global view (downsampled to patch_size) AND local patches. This multi-scale approach helps the model understand both fine details and overall context.

#### `data/custom_transforms.py` - Image Splitting Logic

The `GlobalAndSplitImages` transform (line 105-121) implements a key VLM technique:

```
Original Image (e.g., 1024x512)
         │
         ▼
┌─────────────────────────┐
│   DynamicResize         │  → Resize to max_size, divisible by patch_size
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   GlobalAndSplitImages  │  → Split into patches + create global view
└─────────────────────────┘
         │
         ▼
Output: [global_patch, patch_1, patch_2, patch_3, patch_4]
        (512x512)    (512x512 each from different regions)
```

### Important Criteria to Consider

| Consideration | Why It Matters |
|--------------|----------------|
| GPU Memory | Device selection affects batch size limits |
| Tokenizer Chat Template | Must match the language model's expected format |
| Image Size | Larger images = more patches = more memory |
| Random Seeds | Essential for reproducibility in research |

### Teaching Points

- **Multi-Device Support**: Always write device-agnostic code
- **Tokenizer Special Tokens**: VLMs need extra tokens like `<|image|>` that don't exist in the base LM vocabulary
- **Autoreload for Development**: When modifying source files, autoreload prevents kernel restart

---

## Block 6: Dataloader Creation

### Code
```python
def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size,
                                          vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens,
                              vlm_cfg.lm_chat_template)

    # Load and combine all training datasets
    dataset_names_to_load = train_cfg.train_dataset_name
    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    combined_train_data = []
    for dataset_name in dataset_names_to_load:
        print(f"Loading dataset: {dataset_name}")
        try:
            train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)['train']
            combined_train_data.append(train_ds)
        except Exception as e:
            print(f"Warning: Failed to load dataset config '{dataset_name}'...")
            continue
    train_ds = concatenate_datasets(combined_train_data)
    train_ds = train_ds.shuffle(seed=0)

    # Apply cutoff and split
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    val_ds = train_ds.select(range(train_size, total_samples-1))
    train_ds = train_ds.select(range(train_size))

    # Wrap in custom datasets
    train_dataset = VQADataset(train_ds, tokenizer, image_processor,
                               vlm_cfg.mp_image_token_length)
    val_dataset = VQADataset(val_ds, tokenizer, image_processor,
                             vlm_cfg.mp_image_token_length)

    # Apply sequence packing for efficiency
    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size*4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack
    )

    # Create collators and dataloaders
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    # ... similar for val_loader

    # Warmup dataloaders
    print("Warming up dataloaders...")
    next(iter(train_loader))
    next(iter(val_loader))

    return train_loader, val_loader
```

### What's Happening

This function creates the complete data pipeline:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE FLOW                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HuggingFace Dataset (The Cauldron)                                        │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐                                                       │
│  │ concatenate +   │  Combine multiple sub-datasets                        │
│  │ shuffle         │                                                       │
│  └─────────────────┘                                                       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐                                                       │
│  │ train/val split │  Based on val_ratio (e.g., 80/20)                    │
│  └─────────────────┘                                                       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐   Source: data/datasets.py:107-151                   │
│  │   VQADataset    │   - Process images with image_processor              │
│  │                 │   - Tokenize text with chat template                 │
│  │                 │   - Create loss masks (only train on answers)        │
│  └─────────────────┘                                                       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐   Source: data/advanced_datasets.py:11-237           │
│  │ ConstantLength  │   - Pack variable-length samples into fixed-length   │
│  │    Dataset      │   - Knapsack algorithm for efficient batching        │
│  │                 │   - Background thread for prefetching                │
│  └─────────────────┘                                                       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐   Source: data/collators.py:59-71                    │
│  │  VQACollator    │   - Pad sequences to max_length                      │
│  │                 │   - Use -100 for label padding (ignored in loss)     │
│  │                 │   - Stack tensors for batching                       │
│  └─────────────────┘                                                       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────┐                                                       │
│  │   DataLoader    │   - Batching, shuffling, multi-worker loading        │
│  └─────────────────┘                                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Source Files Deep Dive

#### `data/datasets.py` - VQADataset

The VQADataset class (`data/datasets.py:107-151`) performs several critical operations:

```python
def _process_data(self, item):
    # 1. Process images
    processed_images, splitted_image_counts = self._process_images(images_data)

    # 2. Format messages with image tokens
    messages = self._get_messages(item, splitted_image_counts)

    # 3. Create input_ids and loss mask
    input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)

    # 4. Create labels (shifted for causal LM)
    labels = self._get_labels(input_ids, mask)

    return {
        "images": processed_images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

**Critical Concept - Loss Masking** (`data/datasets.py:145-150`):

```python
def _get_labels(self, input_ids, mask):
    labels = input_ids.clone().masked_fill(~mask, -100)  # Mask non-answer tokens
    labels = labels.roll(-1)  # Shift for next-token prediction
    labels[-1] = -100  # Last token has no target
    return labels
```

This ensures the model only learns to predict assistant responses, not user queries or system prompts.

#### `data/advanced_datasets.py` - ConstantLengthDataset

The ConstantLengthDataset (`data/advanced_datasets.py:11-237`) implements **sequence packing**, a crucial efficiency technique:

**Without Packing:**
```
Batch 1: [sample1 (100 tokens) | PAD PAD PAD ... PAD]  256 tokens
Batch 2: [sample2 (50 tokens)  | PAD PAD PAD ... PAD]  256 tokens
Batch 3: [sample3 (200 tokens) | PAD PAD PAD ... PAD]  256 tokens
→ Total: 768 tokens, but only 350 are real data (45% efficiency)
```

**With Packing (Knapsack Algorithm):**
```
Batch 1: [sample1 (100) | sample2 (50) | sample4 (100) | PAD ...]  256 tokens
Batch 2: [sample3 (200) | sample5 (50) | PAD ...]                  256 tokens
→ Total: 512 tokens, 500 are real data (97% efficiency)
```

The knapsack algorithm (`data/advanced_datasets.py:173-222`) efficiently bins samples:

```python
def _balanced_greedy_knapsack(self, buffer, L, delta=0, max_images_per_knapsack=None):
    # Sort items by length (largest first)
    items = sorted(enumerate(zip(lengths, image_counts)),
                   key=lambda x: x[1][0], reverse=True)

    for idx, (item_len, item_image_count) in items:
        # Find a knapsack that fits both length AND image constraints
        for ks_id in sorted(range(len(knapsack_load)), key=knapsack_load.__getitem__):
            length_fits = knapsack_load[ks_id] + item_len <= L
            image_fits = knapsack_image_counts[ks_id] + item_image_count <= max_images

            if length_fits and image_fits:
                # Add to this knapsack
                knapsack_groups[ks_id].append(idx)
                break
```

### Why This Step is Important

1. **Data Efficiency**: Sequence packing dramatically improves GPU utilization
2. **Memory Management**: `max_images_per_knapsack` prevents OOM errors
3. **Loss Masking**: Only training on answers prevents the model from memorizing questions
4. **Warmup**: Initializes worker processes before training begins

### Important Criteria to Consider

| Parameter | Description | Trade-offs |
|-----------|-------------|------------|
| `data_cutoff_idx` | Limit dataset size | Faster iteration vs. less data |
| `val_ratio` | Train/val split | More val = better monitoring, less training |
| `max_sample_length` | Max tokens per sample | Longer = more context, more memory |
| `seq_length` | Fixed sequence length | Must fit in GPU memory |
| `max_images_per_knapsack` | Images per packed batch | More images = richer context, more memory |
| `num_workers` | DataLoader parallelism | More = faster loading, more CPU/memory |

### Teaching Points

- **The Cauldron Dataset**: A curated mixture of 50+ VQA datasets from Hugging Face
- **Chat Templates**: The `apply_chat_template()` function formats conversations in the expected format for the language model
- **Pin Memory**: `pin_memory=True` speeds up CPU→GPU transfers
- **Persistent Workers**: `persistent_workers=True` keeps workers alive between epochs

---

## Block 7: Training Loop

### Code
```python
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps  # Linear warmup
    if it > max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(train_cfg, vlm_cfg):
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    # Define optimizer groups with different learning rates
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp})
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.vision_encoder.parameters()),
                            'lr': train_cfg.lr_vision_backbone})
    if train_cfg.lr_language_backbone > 0:
        param_groups.append({'params': list(model.decoder.parameters()),
                            'lr': train_cfg.lr_language_backbone})

    optimizer = optim.AdamW(param_groups)
    model.to(device)

    # Training loop
    for i, batch in enumerate(train_loader):
        images = batch["images"]
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

        loss.backward()

        if is_update_step:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
            # Update learning rates with schedule
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(save_directory=vlm_cfg.vlm_checkpoint_path)
    model.push_to_hub(hf_model_name)
```

### What's Happening

This implements the complete training pipeline with several sophisticated techniques.

### Source Files Deep Dive

#### `models/vision_language_model.py` - The VLM Architecture

The VisionLanguageModel class (`models/vision_language_model.py:21-184`) orchestrates three components:

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.vision_encoder = ViT.from_pretrained(cfg)  # SigLIP
        self.decoder = LanguageModel.from_pretrained(cfg)  # SmolLM2
        self.MP = ModalityProjector(cfg)  # Projection layer
```

**Forward Pass** (`models/vision_language_model.py:62-80`):

```python
def forward(self, input_ids, images, attention_mask=None, targets=None):
    # 1. Get text token embeddings
    token_embd = self.decoder.token_embedding(input_ids)  # [B, T_seq, D_lm]

    # 2. Process images through vision encoder + projector
    if images_tensor is not None:
        image_embd = self.vision_encoder(images_tensor)   # [N_img, T_img, D_vit]
        image_embd = self.MP(image_embd)                  # [N_img, 64, D_lm]

        # 3. Replace <|image|> placeholder tokens with image embeddings
        token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

    # 4. Run through language model decoder
    logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

    # 5. Compute loss if training
    if targets is not None:
        logits = self.decoder.head(logits)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              targets.reshape(-1), ignore_index=-100)

    return logits, loss
```

**Key Insight - Token Replacement** (`models/vision_language_model.py:36-49`):

```python
def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
    # Find all positions where input_ids == <|image|> token
    mask = (input_ids == self.tokenizer.image_token_id)
    # Replace those positions with image embeddings
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1))
    return updated_token_embd
```

This is how images get "injected" into the text sequence - placeholder tokens are replaced with actual visual features.

#### `models/modality_projector.py` - Bridging Vision and Language

The ModalityProjector (`models/modality_projector.py:4-45`) performs two key operations:

```python
class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        # Input: vit_hidden_dim * (pixel_shuffle_factor^2) = 768 * 16 = 12,288
        # Output: lm_hidden_dim = 960
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def pixel_shuffle(self, x):
        # Reshape [B, 1024, 768] → [B, 64, 12288]
        # Reduces sequence length by factor of 16, increases embedding dim by 16
        ...

    def forward(self, x):
        x = self.pixel_shuffle(x)  # [B, 1024, 768] → [B, 64, 12288]
        x = self.proj(x)           # [B, 64, 12288] → [B, 64, 960]
        return x
```

**Why Pixel Shuffle?**

```
Vision Transformer Output: 1024 patches × 768 dimensions
        │
        │ (too many tokens for efficient LM processing!)
        ▼
Pixel Shuffle: Combine 4×4 adjacent patches
        │
        ▼
Result: 64 tokens × 12,288 dimensions
        │
        │ (now project to LM space)
        ▼
Linear Projection: 64 tokens × 960 dimensions (matches SmolLM2)
```

This reduces the sequence length from 1024 to 64 tokens (16× reduction), making it feasible to process images alongside text within the language model's context window.

### Learning Rate Schedule

The `get_lr()` function implements a warmup + cosine decay schedule:

```
Learning Rate
     │
max_lr ─────┐
            │╲
            │ ╲
            │  ╲   Cosine Decay
            │   ╲
            │    ╲
min_lr ─────│─────╲─────────────
            │
     │      │
     └──────┴──────────────────→ Steps
         3%        warmup → decay
```

**Why This Schedule?**
- **Warmup**: Prevents early training instability when starting from random (projector) weights
- **Cosine Decay**: Smooth, gradual decrease prevents sudden drops that can destabilize training

### Differential Learning Rates

The notebook uses **three different learning rates**:

| Component | Learning Rate | Reasoning |
|-----------|--------------|-----------|
| Modality Projector (MP) | 0.005 (highest) | Randomly initialized, needs to learn from scratch |
| Vision Backbone | 0.0005 (10× lower) | Pre-trained, fine-tune carefully |
| Language Backbone | 0.0005 (10× lower) | Pre-trained, fine-tune carefully |

```python
param_groups = [
    {'params': model.MP.parameters(), 'lr': 0.005},      # New layer
    {'params': model.vision_encoder.parameters(), 'lr': 0.0005},  # Pre-trained
    {'params': model.decoder.parameters(), 'lr': 0.0005},         # Pre-trained
]
```

**Key Insight**: Pre-trained backbones already have useful representations. We want to adapt them gently while letting the new projection layer learn aggressively.

### Mixed Precision Training

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
```

**Why Mixed Precision?**
- **Memory**: FP16 uses half the memory of FP32
- **Speed**: Modern GPUs have specialized FP16 tensor cores
- **Quality**: `autocast` automatically manages which operations stay in FP32 for numerical stability

### Gradient Accumulation

```python
if train_cfg.gradient_accumulation_steps > 1:
    loss = loss / train_cfg.gradient_accumulation_steps

loss.backward()

if is_update_step:  # Every N steps
    optimizer.step()
    optimizer.zero_grad()
```

**Why Gradient Accumulation?**

With limited GPU memory, you can't always fit a large batch. Gradient accumulation simulates larger batches:

```
Effective Batch Size = batch_size × gradient_accumulation_steps
                     = 1 × 4 = 4 (in the notebook)
```

### Important Criteria to Consider

| Hyperparameter | Impact | Tuning Tips |
|----------------|--------|-------------|
| `lr_mp` | Projector learning speed | Higher = faster convergence, risk of instability |
| `lr_backbone` | Backbone adaptation | Lower preserves pre-trained features |
| `gradient_accumulation_steps` | Effective batch size | Increase if GPU memory is limited |
| `max_grad_norm` | Gradient clipping | 1.0 is typical; lower for stability |
| `max_training_steps` | Total training duration | More steps = better, but diminishing returns |

### Teaching Points

- **Transfer Learning**: The backbones (ViT, SmolLM2) are pre-trained; we're fine-tuning for VLM
- **Parameter Efficiency**: Only the projector is trained from scratch (very few parameters)
- **Gradient Clipping**: Prevents exploding gradients, especially important with mixed precision

---

## Block 8: Configuration Classes

### Code
```python
@dataclass
class VLMConfig:
    # Vision Transformer (ViT) Configuration
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim  # FFN intermediate dimension
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_n_blocks: int = 12
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    # Language Model Configuration
    lm_hidden_dim: int = 960
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_n_blocks: int = 32
    lm_max_length: int = 256
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}..."

    # Modality Projector Configuration
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    # Special tokens for VLM
    vlm_extra_tokens: dict = field(default_factory=lambda: {
        "image_token": "<|image|>",
        "global_image_token": "<|global_image|>",
        "r1c1": "<row_1_col_1>", ...  # 64 position tokens
    })

@dataclass
class TrainConfig:
    lr_mp: float = 0.005
    lr_vision_backbone: float = 0.0005
    lr_language_backbone: float = 0.0005
    data_cutoff_idx: int = 128  # Small subset for testing
    val_ratio: float = 0.2
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_training_steps: int = 200
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("tqa", )
```

### What's Happening

These dataclasses centralize all hyperparameters for the model architecture and training process.

### Source Files Referenced

The notebook defines its own configs that mirror `models/config.py`, with modifications for Colab's limited resources.

### Architecture Configuration Breakdown

#### Vision Transformer (SigLIP)

```
┌─────────────────────────────────────────────────────────────┐
│                      SigLIP-B/16-512                         │
├─────────────────────────────────────────────────────────────┤
│  Input: 512×512 image                                        │
│      │                                                       │
│      ▼                                                       │
│  Patch Embedding: 16×16 patches → 32×32 = 1024 patches      │
│      │                                                       │
│      ▼                                                       │
│  12 Transformer Blocks (hidden_dim=768, 12 heads)           │
│      │                                                       │
│      ▼                                                       │
│  Output: [1024, 768] patch embeddings                       │
└─────────────────────────────────────────────────────────────┘
```

**Why SigLIP?**
- Trained with a sigmoid loss (better for retrieval tasks)
- Strong performance on visual understanding benchmarks
- Efficient patch-based representation

#### Language Model (SmolLM2)

```
┌─────────────────────────────────────────────────────────────┐
│                       SmolLM2-135M                           │
├─────────────────────────────────────────────────────────────┤
│  Vocabulary: 49,152 + 66 special tokens = 49,218            │
│  Hidden Dimension: 960                                       │
│  Attention Heads: 15 (with 5 KV heads - GQA)                │
│  Layers: 32                                                  │
│  Context Length: 256 (notebook) / 8192 (full)               │
└─────────────────────────────────────────────────────────────┘
```

**Why SmolLM2?**
- Small enough to train on consumer GPUs
- Surprisingly capable despite size
- Instruction-tuned variant available for better chat formatting

#### Modality Projector Math

```
Input:  1024 patches × 768 dimensions (from ViT)

Pixel Shuffle (factor=4):
        ├─ Reshape: [1024, 768] → [32, 32, 768]
        ├─ Group 4×4 patches: [8, 8, 768×16]
        └─ Flatten: [64, 12288]

Linear Projection:
        [64, 12288] → [64, 960]

Output: 64 image tokens × 960 dimensions (matches LM)
```

**mp_image_token_length = 64** means each image becomes 64 tokens in the language model's context.

#### Special Tokens Explained

```python
vlm_extra_tokens = {
    "image_token": "<|image|>",           # Placeholder replaced with image embeddings
    "global_image_token": "<|global_image|>",  # Marks the global view
    "r1c1": "<row_1_col_1>", ...          # Position markers for split image patches
}
```

These 66 tokens are added to the tokenizer's vocabulary to handle multi-image, spatially-aware inputs.

### Training Configuration Explained

| Parameter | Notebook Value | Production Value | Why Different? |
|-----------|---------------|------------------|----------------|
| `data_cutoff_idx` | 128 | None | Testing vs. full training |
| `batch_size` | 1 | 2+ | Colab memory limits |
| `max_training_steps` | 200 | 40,000 | Quick demo vs. convergence |
| `lm_max_length` | 256 | 4096 | Memory vs. context |
| `train_dataset_name` | ("tqa",) | ("all",) | Single dataset for testing |

### Important Criteria to Consider

#### Model Size Trade-offs

| Aspect | Smaller Model | Larger Model |
|--------|--------------|--------------|
| Training Speed | Faster | Slower |
| GPU Memory | Lower | Higher |
| Capability | Limited | Better |
| Inference Speed | Faster | Slower |

#### Context Length Considerations

```
lm_max_length = 256 (notebook)

Available for text: 256 - (64 × num_images) = 256 - 64 = 192 tokens

With 2 images: 256 - 128 = 128 tokens for text
```

**Trade-off**: More images = less room for text. The 256 limit in the notebook is very constrained.

### Teaching Points

- **Dataclasses**: Python's `@dataclass` decorator auto-generates `__init__`, `__repr__`, etc.
- **Grouped Query Attention**: `lm_n_kv_heads=5` (vs 15 attention heads) reduces memory with minimal quality loss
- **Weight Tying**: `lm_tie_weights=True` shares embedding and output projection weights

---

## Block 9: Run Training

### Code
```python
vlm_cfg = VLMConfig()
train_cfg = TrainConfig()
train(train_cfg, vlm_cfg)
```

### What's Happening

1. Instantiate configuration objects with notebook-specific values
2. Call the `train()` function which:
   - Creates dataloaders
   - Initializes the VisionLanguageModel
   - Runs the training loop
   - Saves checkpoints
   - Pushes to Hugging Face Hub

### Expected Output

```
Loading dataset: tqa
Warming up dataloaders...
Warmup complete.
Loading from backbone weights
nanoVLM initialized with 222,000,000 parameters
Training summary: 102 samples, 25 batches/epoch, batch size 1
Starting training loop

Step: 0, Loss: 4.2341, Val Loss: 4.1892, Tokens/s: 125.32
Step: 20, Loss: 3.8921, Val Loss: 3.7654, Tokens/s: 142.18
...
Step: 180, Loss: 2.1234, Val Loss: 2.0987, Tokens/s: 145.67

Epoch 1 | Train Loss: 2.8765 | Val Loss: 2.7890 | Time: 45.23s | T/s: 138.45
Total training time: 180.92s
```

### What to Monitor

1. **Loss Curves**: Should decrease over time
   - Training loss: Directly optimized
   - Validation loss: Generalization indicator

2. **Tokens/second**: Measures training efficiency
   - Higher = better GPU utilization
   - Varies with batch size and sequence length

3. **Overfitting Signs**:
   - Training loss decreases but validation increases
   - Gap between train and val loss grows

### Teaching Points

- **Quick Iteration**: The notebook's small settings allow rapid experimentation
- **Scaling Up**: For real training, increase `data_cutoff_idx`, `max_training_steps`, and use "all" datasets
- **Checkpointing**: Models are saved both locally and to HF Hub

---

## Summary: Key Concepts for Teaching VLMs

### 1. The Three Components

```
Vision Encoder → Modality Projector → Language Model
   (SigLIP)          (MLP + Pixel Shuffle)    (SmolLM2)
```

### 2. The Key Innovation: Token Replacement

Images are converted to embeddings and inserted at `<|image|>` placeholder positions in the text sequence.

### 3. Training Efficiency Techniques

- **Sequence Packing**: Knapsack algorithm for batching
- **Mixed Precision**: FP16 for speed and memory
- **Gradient Accumulation**: Simulate larger batches
- **Differential Learning Rates**: Higher LR for new layers

### 4. Data Pipeline

```
Raw Data → VQADataset → ConstantLengthDataset → VQACollator → DataLoader
```

### 5. Loss Masking

Only compute loss on assistant responses, not user queries or image tokens.

---

## Further Reading

- **Paper**: "SigLIP: Sigmoid Loss for Language-Image Pre-Training"
- **SmolLM2**: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- **The Cauldron Dataset**: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
- **nanoVLM Repository**: https://github.com/huggingface/nanoVLM

---

## Exercises for Students

1. **Change the Vision Backbone**: Modify `vit_model_type` to use a different ViT and observe performance changes

2. **Experiment with Learning Rates**: Try freezing the vision backbone (`lr_vision_backbone=0`) and compare results

3. **Add More Datasets**: Change `train_dataset_name` to include multiple datasets from The Cauldron

4. **Analyze Attention**: Add code to visualize which image regions the model attends to

5. **Benchmark Inference**: Use `generate.py` to test the trained model on custom images
