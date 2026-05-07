import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torchvision import transforms




NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]

# CLIP normalisation statistics (ViT models)
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

IMAGE_SIZE = (224, 224)   # CLIP ViT-B/16 input resolution



def load_clip_model(device: str = "cpu"):
    """Load CLIP ViT-B/16 (the 'victim' model in the paper)."""
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    return model, preprocess


def build_text_features(
    class_names: list,
    clip_model,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode zero-shot text prompts and return normalised feature matrix.

    Returns
    -------
    text_features : Tensor [num_classes, embed_dim]
    """
    prompts = [f"a photo of a {c}" for c in class_names]
    tokens  = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)



def load_mask_from_file(
    mask_path: str,
    image_size: tuple = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Load a binary PNG mask (white = object, black = background).

    Returns
    -------
    mask : Tensor [1, 3, H, W]  float32, values in {0, 1}
    """
    mask_img = Image.open(mask_path).convert("L").resize(image_size, Image.NEAREST)
    mask_np  = (np.array(mask_img) > 127).astype(np.float32)          # [H, W]
    mask     = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)     # [1,1,H,W]
    return mask.expand(1, 3, *image_size)                              # [1,3,H,W]


def make_full_mask(image_size: tuple = IMAGE_SIZE) -> torch.Tensor:
    """Full-image mask (no masking — entire image perturbed)."""
    return torch.ones(1, 3, *image_size)



def m_ifgsm(
    image_tensor:    torch.Tensor,   # [1, 3, H, W]  float32, pixel range [0, 255]
    mask:            torch.Tensor,   # [1, 3, H, W]  binary float {0, 1}
    clip_model,
    true_label_idx:  int,            # ground-truth class index
    text_features:   torch.Tensor,   # [num_classes, D]  normalised CLIP text embeddings
    epsilon:         float = 8.0,    # per-step magnitude in pixel space (paper uses small ε)
    num_iterations:  int   = 100,    # max iterations N
    loss_threshold:  float = 20.0,   # τ for early stopping
    device:          str   = "cpu",
    targeted:        bool  = False,
    target_label_idx: int  = None,   # required when targeted=True
    verbose:         bool  = True,
) -> torch.Tensor:
    assert not targeted or target_label_idx is not None, \
        "target_label_idx must be provided for targeted attacks."

    image_tensor = image_tensor.to(device).float()
    mask         = mask.to(device).float()
    text_features = text_features.to(device)

    # Background-only image (never modified)
    X_inv = image_tensor * (1.0 - mask)

    # Shared CLIP normalisation transform
    clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    X_adv = image_tensor.clone()

    for n in range(num_iterations):

        # ── Enable gradients for this step ──
        X_adv = X_adv.detach().requires_grad_(True)

        # ── Forward pass through CLIP image encoder ──
        img_01   = X_adv / 255.0                              # scale to [0, 1]
        img_norm = clip_norm(img_01)                           # CLIP normalisation
        image_features = clip_model.encode_image(img_norm)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity logits  →  softmax probabilities
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)   # [1, C]
        probs  = F.softmax(logits, dim=-1)                          # [1, C]

        # ── Loss ──
        if targeted:
            label = torch.tensor([target_label_idx], device=device)
        else:
            label = torch.tensor([true_label_idx],   device=device)

        loss = F.cross_entropy(logits, label)

        # ── Early stopping (Algorithm 1, line 8) ──
        true_prob = probs[0, true_label_idx].item()
        if true_prob == 0.0 and loss.item() > loss_threshold:
            if verbose:
                print(f"    Early stop @ iter {n}  loss={loss.item():.2f}")
            break

        # ── Backprop ──
        loss.backward()
        grad = X_adv.grad.detach()

        # ── Masked gradient-sign update ──
        sign = grad.sign()
        with torch.no_grad():
            if targeted:
                # Eq. 2: subtract gradient → push toward target
                X_new = X_inv + mask * (X_adv.detach() - epsilon * sign)
            else:
                # Eq. 1: add gradient → push away from true class
                X_new = X_inv + mask * (X_adv.detach() + epsilon * sign)

            # Clip pixel values to valid RGB range
            X_adv = torch.clamp(X_new, 0.0, 255.0)

        if verbose and (n + 1) % 20 == 0:
            print(f"    iter {n+1:>4}/{num_iterations}  "
                  f"loss={loss.item():.4f}  "
                  f"p(true)={true_prob:.4f}")

    return X_adv.detach()


def load_nerf_image_paths(object_dir: str, split: str = "train") -> list:
    
    json_path = os.path.join(object_dir, f"transforms_{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"transforms file not found: {json_path}")

    with open(json_path) as f:
        meta = json.load(f)

    paths = []
    for i, frame in enumerate(meta.get("frames", [])):
        rel = frame["file_path"].lstrip("./")          # strip leading "./"
        for ext in [".png", ".jpg", ".jpeg", ""]:
            candidate = os.path.join(object_dir, rel + ext)
            if os.path.exists(candidate):
                paths.append((candidate, i))
                break
        else:
            print(f"  Warning: image not found for frame {i}: {rel}")

    return paths



def mirror_object_structure(object_dir: str, out_object_dir: str) -> None:
    import shutil

    os.makedirs(out_object_dir, exist_ok=True)

    # Copy every file at the object root (JSON, PLY, etc.) except images
    image_exts = {".png", ".jpg", ".jpeg"}
    for fname in os.listdir(object_dir):
        src = os.path.join(object_dir, fname)
        if os.path.isfile(src) and Path(fname).suffix.lower() not in image_exts:
            dst = os.path.join(out_object_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Create split sub-folders so 3DGS tools never complain about missing dirs
    for split in ["train", "test", "val"]:
        split_src = os.path.join(object_dir, split)
        if os.path.isdir(split_src):
            os.makedirs(os.path.join(out_object_dir, split), exist_ok=True)


def process_object_split(
    object_dir:     str,
    object_name:    str,
    out_object_dir: str,           # <output_root>/<object_name>  (already created)
    clip_model,
    text_features:  torch.Tensor,
    class_names:    list,
    split:          str   = "train",
    epsilon:        float = 8.0,
    num_iterations: int   = 100,
    loss_threshold: float = 20.0,
    mask_dir:       str   = None,
    device:         str   = "cpu",
    verbose:        bool  = True,
) -> list:
    true_label_idx = class_names.index(object_name)
    image_paths    = load_nerf_image_paths(object_dir, split)

    out_split_dir  = os.path.join(out_object_dir, split)
    os.makedirs(out_split_dir, exist_ok=True)

    # Resize only for CLIP (internal); we keep the original resolution for saving
    clip_resize = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),   # → [0, 1]
    ])
    to_tensor_orig = transforms.ToTensor()   # no resize

    results = []
    total   = len(image_paths)

    print(f"\n  [{object_name}/{split}]  {total} images  "
          f"ε={epsilon}  N={num_iterations}")

    for idx, (img_path, frame_idx) in enumerate(image_paths):
        filename = Path(img_path).name   # e.g. "r_0.png"  — kept exactly
        stem     = Path(img_path).stem   # e.g. "r_0"

        if verbose:
            print(f"    ({idx+1}/{total}) {filename}")

        # ── Load image, handle RGBA ──────────────────────────────────────
        img_pil  = Image.open(img_path)
        has_alpha = img_pil.mode == "RGBA"
        if has_alpha:
            alpha_channel = img_pil.split()[3]   # save alpha for later
        img_rgb  = img_pil.convert("RGB")

        orig_w, orig_h = img_rgb.size   # remember original resolution

        # Tensor at CLIP resolution for the attack
        # Move to device immediately so all ops stay on the same device
        img_clip_tensor = (clip_resize(img_rgb).unsqueeze(0) * 255.0).to(device)  # [1,3,224,224]

        # Load or construct mask (at CLIP resolution), always on device
        mask = None
        if mask_dir:
            for mname in [f"{stem}_mask.png", f"{stem}.png"]:
                mp = os.path.join(mask_dir, object_name, split, mname)
                if os.path.exists(mp):
                    mask = load_mask_from_file(mp, IMAGE_SIZE).to(device)
                    break
            if mask is None and verbose:
                print(f"      mask not found - using full-image perturbation")

        if mask is None:
            mask = make_full_mask(IMAGE_SIZE).to(device)

        # Run M-IFGSM at CLIP resolution
        adv_clip = m_ifgsm(
            image_tensor    = img_clip_tensor,
            mask            = mask,
            clip_model      = clip_model,
            true_label_idx  = true_label_idx,
            text_features   = text_features,
            epsilon         = epsilon,
            num_iterations  = num_iterations,
            loss_threshold  = loss_threshold,
            device          = device,
            verbose         = verbose,
        )   # [1, 3, 224, 224] float32 in [0, 255], on `device`

        # Compute perturbation noise at CLIP resolution, then upsample to
        # the original image size and add it to the full-resolution image.
        # All tensors are kept on `device` until the final .cpu() call.
        noise_clip = adv_clip - img_clip_tensor   # [1, 3, 224, 224]  on device

        # Upsample noise to original image dimensions (still on device)
        noise_orig = torch.nn.functional.interpolate(
            noise_clip,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )   # [1, 3, H, W]  on device

        # Full-resolution original tensor - send to same device
        orig_tensor = (to_tensor_orig(img_rgb).unsqueeze(0) * 255.0).to(device)  # [1,3,H,W]

        # Add noise, clip to [0,255], then move back to CPU for PIL conversion
        adv_orig = torch.clamp(orig_tensor + noise_orig, 0.0, 255.0).cpu()

        # ── Convert back to PIL and save with ORIGINAL filename ──────────
        adv_np  = adv_orig.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        adv_pil = Image.fromarray(adv_np, mode="RGB")

        if has_alpha:
            # Re-attach original alpha channel
            adv_pil = adv_pil.convert("RGBA")
            adv_pil.putalpha(alpha_channel)

        out_path = os.path.join(out_split_dir, filename)   # same filename!
        adv_pil.save(out_path)

        results.append({"frame": frame_idx, "orig_path": img_path, "adv_path": out_path})

    return results


# Keep backward-compatible alias
def process_object(
    object_dir:    str,
    object_name:   str,
    output_dir:    str,
    clip_model,
    text_features: torch.Tensor,
    class_names:   list,
    split:         str   = "train",
    epsilon:       float = 8.0,
    num_iterations: int  = 100,
    loss_threshold: float = 20.0,
    mask_dir:      str   = None,
    device:        str   = "cpu",
    verbose:       bool  = True,
) -> list:
    """Wrapper kept for compatibility — delegates to process_object_split."""
    out_object_dir = os.path.join(output_dir, object_name)
    mirror_object_structure(object_dir, out_object_dir)
    return process_object_split(
        object_dir     = object_dir,
        object_name    = object_name,
        out_object_dir = out_object_dir,
        clip_model     = clip_model,
        text_features  = text_features,
        class_names    = class_names,
        split          = split,
        epsilon        = epsilon,
        num_iterations = num_iterations,
        loss_threshold = loss_threshold,
        mask_dir       = mask_dir,
        device         = device,
        verbose        = verbose,
    )