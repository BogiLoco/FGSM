"""
FGSM (Fast Gradient Sign Method) - local experimentation playground.

Includes:
- untargeted FGSM  (we want to INCREASE the loss with respect to the true label)
- targeted FGSM    (we want to DECREASE the loss with respect to the target label)
- visualization of the original vs. the attacked image
- epsilon sweep across different perturbation budgets

Model: a ready, pretrained resnet18 from torchvision (ImageNet), so you don't
need to train anything yourself - you can start attacking a real, "strong"
model right away.

Requirements:
    pip install torch torchvision matplotlib pillow requests
    
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Load a ready-made, pretrained model (ImageNet, 1000 classes)
# ---------------------------------------------------------------------------

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # eval mode - disables dropout/batchnorm-training behavior etc.
model.to(DEVICE)

# IMPORTANT: freeze the model weights - we don't want to train the model,
# we only want to compute the gradient WITH RESPECT TO THE IMAGE (not the weights!)
for param in model.parameters():
    param.requires_grad = False

# ImageNet label list (for printing readable class names instead of raw indices)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(LABELS_URL).text.strip().split("\n")


# ---------------------------------------------------------------------------
# 2. Loading and preprocessing the image
# ---------------------------------------------------------------------------

# Standard resnet/ImageNet normalization - mean/std computed over the entire
# ImageNet training set; that's what the model was trained on.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # PIL Image [0-255] -> float tensor [0,1]
    ]
)


def load_image(path_or_url: str) -> torch.Tensor:
    """Returns an image tensor in [0,1] range, shape [3, 224, 224]."""
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return preprocess(img)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalization for the model's input (the model expects normalized data)."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def predict(x_01: torch.Tensor) -> tuple[int, float]:
    """
    x_01: image tensor in [0,1] range (NOT normalized).
    Returns (class_index, confidence).
    """
    with torch.no_grad():
        x_norm = normalize(x_01)
        logits = model(x_norm.unsqueeze(0).to(DEVICE))
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return pred_idx.item(), conf.item()


# ---------------------------------------------------------------------------
# 3. UNTARGETED FGSM
#    Goal: make the model misclassify the image as "anything else" - we
#    don't care about a specific wrong class, just that it loses the true one.
# ---------------------------------------------------------------------------

def fgsm_untargeted(x_01: torch.Tensor, true_label: int, epsilon: float) -> torch.Tensor:
    """
    x_01:       image [3,H,W] in [0,1] range (UNNORMALIZED - the "real" pixel
                scale, where it makes sense to talk about epsilon in L_inf)
    true_label: the true label (ImageNet class index)
    epsilon:    L_inf perturbation budget, e.g. 8/255 - images on disk are
                stored as uint8 [0,255], so epsilon is expressed in that same
                scale and divided by 255, so it represents the same physical
                brightness change after normalizing the image to [0,1]

    Returns: the attacked image, in [0,1] range, as a "dead" tensor on the CPU
             (ready for display/saving, with no trace left in the autograd graph)
    """
    # .clone() - don't modify the original tensor passed in by the caller
    # .detach() - on input: defensively cut off any existing graph, in case
    #             x_01 was the result of a previous attack (e.g. when chaining
    #             FGSM calls) rather than a fresh tensor from load_image().
    #             A fresh tensor from load_image() is already a clean graph
    #             leaf with requires_grad=False, so this line is a no-op in
    #             that case - but the function stays correct regardless of
    #             where x_01 actually came from
    x_01 = x_01.clone().detach().to(DEVICE)
    x_01.requires_grad_(True)  # we want the gradient WITH RESPECT TO PIXELS, not model weights

    # normalize() - a "translator" between the attack space (raw pixels [0,1],
    # where epsilon has a physical meaning) and the space the MODEL expects
    # (mean=0, std=1 per channel - because that's how it was trained on ImageNet).
    # This is NOT part of the FGSM algorithm itself - it's an adaptation to
    # this specific model/training pipeline (e.g. a simple CNN trained on
    # MNIST without normalization wouldn't have this step at all)
    x_norm = normalize(x_01)

    # unsqueeze(0) - the model expects a batch dimension [B, C, H, W]; even
    # for a single image you must artificially add B=1, otherwise the forward
    # pass throws an error (the network's layers are written for 4D input)
    logits = model(x_norm.unsqueeze(0))

    # loss with respect to the TRUE label - we want to INCREASE it (the model should
    # misclassify "in any direction" - we don't care about a specific wrong
    # class, hence "untargeted")
    loss = F.cross_entropy(logits, torch.tensor([true_label]).to(DEVICE))

    model.zero_grad()
    loss.backward()  # computes dloss/dx_01 (because requires_grad=True was set on x_01)

    # sign() - we take ONLY the direction of the gradient (-1, 0, +1 per
    # pixel), not its magnitude. This is the optimal solution to "maximize
    # grad·perturbation subject to ||perturbation||_inf <= epsilon" -
    # every pixel gets the MAXIMUM allowed step in its own direction, so we
    # fully exploit the L_inf budget in a single step
    #
    # NOTE: x_01 itself still has requires_grad=True (it's a leaf tensor,
    # set explicitly above) - the fact that grad_sign has no graph trace is
    # NOT enough on its own. "x_01 + anything" still inherits
    # requires_grad=True from x_01, so building x_adv must explicitly detach
    # from the graph - hence torch.no_grad()
    with torch.no_grad():
        # UNTARGETED: move IN THE DIRECTION of the gradient (+epsilon) ->
        # increases the loss with respect to the true label, so the model "believes"
        # in it less and less
        x_adv = x_01 + epsilon * x_01.grad.sign()

        # clamp to [0,1] - the image must stay within a valid, displayable
        # pixel range (otherwise e.g. 0.98 + epsilon could give 1.02)
        x_adv = torch.clamp(x_adv, 0, 1)

    # .cpu() - if DEVICE="cuda", the data lives in GPU memory, and
    # matplotlib/PIL/numpy can only read from CPU memory (RAM), so it needs
    # to be physically moved there before further use (on a machine without
    # a GPU this is a plain no-op, nothing breaks)
    return x_adv.cpu()


# ---------------------------------------------------------------------------
# 4. TARGETED FGSM
#    Goal: make the model misclassify the image as SPECIFICALLY the chosen
#    class (e.g. label a "cat" as a "toaster").
# ---------------------------------------------------------------------------

def fgsm_targeted(x_01: torch.Tensor, target_label: int, epsilon: float) -> torch.Tensor:
    """
    x_01:         image [3,H,W] in [0,1] range (unnormalized)
    target_label: the label we want to "convince" the model of (class index) -
                  the only conceptual difference from untargeted: we plug in
                  a FALSE label instead of the true one as the second argument
                  to the exact same loss function (cross-entropy)
    epsilon:      L_inf perturbation budget

    Returns: the attacked image, in [0,1] range
    """
    # see the comments in fgsm_untargeted() - identical logic:
    # defensive clone().detach() on input, requires_grad only on x_01
    x_01 = x_01.clone().detach().to(DEVICE)
    x_01.requires_grad_(True)

    x_norm = normalize(x_01)  # "translator" into the space the model expects
    logits = model(x_norm.unsqueeze(0))  # unsqueeze(0) = artificial batch dimension

    # loss with respect to the TARGET (false) label - we want to DECREASE it (the
    # smaller the loss with respect to target_label, the more the model "believes"
    # that's the correct class - this is the same cross-entropy as in
    # untargeted, only the label plugged into it changes)
    loss = F.cross_entropy(logits, torch.tensor([target_label]).to(DEVICE))

    model.zero_grad()
    loss.backward()

    # x_01 itself still has requires_grad=True (leaf tensor), so building
    # x_adv must be wrapped in torch.no_grad() - same reasoning as in
    # fgsm_untargeted() above
    with torch.no_grad():
        # TARGETED: move AGAINST the direction of the gradient (-epsilon) ->
        # decreases the loss with respect to target_label, so the model increasingly
        # "believes" in the false class. This is the only code difference
        # from untargeted: the sign in front of epsilon (+ for untargeted,
        # - for targeted)
        x_adv = x_01 - epsilon * x_01.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)  # back to a valid pixel range

    return x_adv.cpu()  # move from GPU to CPU for further use (plotting etc.)


# ---------------------------------------------------------------------------
# 5. Visualization - side-by-side comparison of original vs. attacked image
# ---------------------------------------------------------------------------

def show_comparison(x_orig: torch.Tensor, x_adv: torch.Tensor, title: str = ""):
    orig_idx, orig_conf = predict(x_orig)
    adv_idx, adv_conf = predict(x_adv)

    # console prints - independent of whether the matplotlib window actually
    # gets displayed (plt.show() blocks execution until the window is closed,
    # so without these prints your terminal log would only show whatever was
    # explicitly print()-ed outside this function, e.g. in epsilon_sweep())
    print(f"\n[{title}]")
    print(f"  before attack: {imagenet_classes[orig_idx]:<25} ({orig_conf:.2%})")
    print(f"  after attack:  {imagenet_classes[adv_idx]:<25} ({adv_conf:.2%})")

    # difference visualization, rescaled to [0,1] purely FOR DISPLAY
    # (so you can see WHAT changed, not the actual raw values)
    diff = (x_adv - x_orig)
    diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(x_orig.permute(1, 2, 0))
    axes[0].set_title(f"Original\n{imagenet_classes[orig_idx]} ({orig_conf:.2%})")
    axes[0].axis("off")

    axes[1].imshow(diff_vis.permute(1, 2, 0))
    axes[1].set_title("Perturbation (visually amplified)")
    axes[1].axis("off")

    axes[2].imshow(x_adv.permute(1, 2, 0))
    axes[2].set_title(f"After attack\n{imagenet_classes[adv_idx]} ({adv_conf:.2%})")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Epsilon sweep - to see from which value the attack starts working
# ---------------------------------------------------------------------------

def epsilon_sweep(x_01: torch.Tensor, true_label: int, epsilons=None):
    if epsilons is None:
        epsilons = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255]

    print(f"{'epsilon':>10} | {'prediction':<25} | {'confidence':>9}")
    print("-" * 52)
    for eps in epsilons:
        if eps == 0:
            x_adv = x_01
        else:
            x_adv = fgsm_untargeted(x_01, true_label, eps)
        idx, conf = predict(x_adv)
        marker = "  <- attack successful!" if idx != true_label else ""
        print(f"{eps:>10.4f} | {imagenet_classes[idx]:<25} | {conf:>9.2%}{marker}")


def epsilon_sweep_targeted(x_01: torch.Tensor, target_label: int, epsilons=None):
    """
    Analogous to epsilon_sweep(), but for the TARGETED attack - checks from
    which epsilon the model starts believing a SPECIFIC, chosen class
    (target_label), rather than just "anything other than the truth".

    Example usage ("make the dog look like a toaster to the model"):
        epsilon_sweep_targeted(x, target_label=859)  # 859 = toaster
    """
    print(f"Attack target: {imagenet_classes[target_label]!r} (index {target_label})\n")
    print(f"{'epsilon':>10} | {'prediction':<25} | {'confidence':>9}")
    print("-" * 52)

    if epsilons is None:
        epsilons = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255, 64 / 255]

    for eps in epsilons:
        if eps == 0:
            x_adv = x_01
        else:
            x_adv = fgsm_targeted(x_01, target_label, eps)

        idx, conf = predict(x_adv)

        # targeted success = the model doesn't just misclassify, it
        # misclassifies EXACTLY in the direction we chose (idx == target_label),
        # unlike untargeted, where any misclassification counts (idx != true_label)
        success = idx == target_label
        marker = "  <- SUCCESS: model now sees the target!" if success else ""
        print(f"{eps:>10.4f} | {imagenet_classes[idx]:<25} | {conf:>9.2%}{marker}")


# ---------------------------------------------------------------------------
# 7. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # example public image (replace with your own local file: "my_image.jpg")
    IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    x = load_image(IMAGE_URL)

    true_idx, true_conf = predict(x)
    print(f"Model's true prediction: {imagenet_classes[true_idx]} ({true_conf:.2%})")

    # --- UNTARGETED ---
    EPS = 8 / 255
    x_adv_untargeted = fgsm_untargeted(x, true_label=true_idx, epsilon=EPS)
    show_comparison(x, x_adv_untargeted, title=f"FGSM untargeted, eps={EPS:.3f}")

    # --- TARGETED ---
    # pick any other class from imagenet_classes, e.g. 859 = "toaster"
    TARGET_CLASS_IDX = 859
    x_adv_targeted = fgsm_targeted(x, target_label=TARGET_CLASS_IDX, epsilon=EPS)
    show_comparison(
        x, x_adv_targeted,
        title=f"FGSM targeted -> {imagenet_classes[TARGET_CLASS_IDX]}, eps={EPS:.3f}",
    )

    # --- SWEEP: from which epsilon does the attack start working? ---
    print("\nEpsilon sweep (untargeted):")
    epsilon_sweep(x, true_label=true_idx)

    # --- SWEEP: from which epsilon does "the dog start looking like a toaster"? ---
    print("\nEpsilon sweep (targeted -> toaster):")
    epsilon_sweep_targeted(x, target_label=TARGET_CLASS_IDX)