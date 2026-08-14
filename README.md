# FGSM — Fast Gradient Sign Method Playground

A local playground for the **Fast Gradient Sign Method (FGSM)** — a white-box
adversarial attack against image classifiers, demonstrated here against a pretrained
ResNet18 (ImageNet). Point it at an image, watch a tiny, human-invisible pixel
perturbation make a confident classifier confidently wrong.

---

## Contents

- [What's in here](#whats-in-here)
- [Model & data](#model--data)
- [Setup](#setup)
- [Usage](#usage)
- [Using your own image](#using-your-own-image)
- [Note](#note)

---

## What's in here

| Feature | Description |
|---|---|
| **Untargeted FGSM** | Perturb an image just enough to make the model misclassify it as *anything other than* the true label. |
| **Targeted FGSM** | Perturb an image to make the model misclassify it as a *specific* chosen class (e.g. "make the model see a toaster instead of a dog"). |
| **Visualization** | Side-by-side original / perturbation / attacked image via `matplotlib`. |
| **Epsilon sweep** | Runs the attack across a range of perturbation budgets to show at which point it starts succeeding. |

No training required — the model is a ready, pretrained `resnet18` from
`torchvision`, so the script attacks a real, already-strong classifier out of the box.

---

## Model & data

| | |
|---|---|
| **Network** | `resnet18` from `torchvision.models`, loaded with `ResNet18_Weights.IMAGENET1K_V1` — pretrained on ImageNet-1K (1000 classes, ~1.28M training images). No local training happens; weights download automatically on first run and are cached under `~/.cache/torch/hub/checkpoints/`. |
| **Class labels** | Human-readable ImageNet class names (e.g. `"golden retriever"` instead of index `207`), fetched at runtime from [`pytorch/hub/imagenet_classes.txt`](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt). |
| **Input images** | Not bundled — `load_image()` accepts a local file path *or* an image URL. The example uses a sample dog photo from the [pytorch/hub repo](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg). Any RGB image works. |
| **Full dataset (optional)** | If you want to test against many images systematically rather than one at a time, the full ImageNet dataset is available at [image-net.org](https://image-net.org) (registration required) — not needed to run these scripts. |

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install torch torchvision matplotlib pillow requests
```

## Usage

```bash
python fgsm.py
```

Runs the example in `if __name__ == "__main__":`:

1. downloads a sample dog image
2. runs an **untargeted** attack
3. runs a **targeted** attack (toward `"toaster"`)
4. runs an **epsilon sweep** for both, printing a table of "how much perturbation
   until the attack succeeds"

## Using your own image

There's no CLI flag (yet) — swap the image by editing one line near the bottom of
the script:

```python
# fgsm.py, inside if __name__ == "__main__":
IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
```

Replace it with either:
- a **local path**: `IMAGE_URL = "my_image.jpg"`
- a **different URL**: any direct link to a `.jpg`/`.png`

`load_image()` handles both transparently. To target a different class in the
targeted attack, change `TARGET_CLASS_IDX` a few lines below (any index 0–999 —
see the [class label list](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)
for what each index means).

---

## Note

This attacks a local model instance you control, on images you provide — it's an
educational/research tool for understanding adversarial robustness, not something
aimed at a live third-party system.
