import os
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.resnet import ResNet
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))

class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class SEBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        reduction=16,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("SEBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SEBasicBlock")

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.se = SEBlock(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


def make_sebasicblock(reduction: int):

    class SEBasicBlockFixed(SEBasicBlock):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     groups=1, base_width=64, dilation=1, norm_layer=None):
            super().__init__(inplanes, planes, stride, downsample,
                             groups, base_width, dilation, norm_layer,
                             reduction=reduction)
    return SEBasicBlockFixed


def seresnet18(num_classes: int, reduction: int = 16):

    block = make_sebasicblock(reduction)
    return ResNet(block, layers=[2, 2, 2, 2], num_classes=num_classes)


def load_model(checkpoint_path: str, device: torch.device):

    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    reduction = checkpoint.get('cfg', {}).get('se_reduction', 16)

    model = seresnet18(num_classes=num_classes, reduction=reduction)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")

    return model, class_names


def get_transform():

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def predict_image(model, image_path: str, transform, device: torch.device, class_names):

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        'image': image,
        'predicted_class': class_names[pred_idx],
        'confidence': confidence,
        'all_probs': probs[0].cpu().numpy()
    }


def visualize_prediction(result, class_names, save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(result['image'])
    ax1.axis('off')
    title = f"Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.2%}"
    ax1.set_title(title, fontsize=14, fontweight='bold')

    probs = result['all_probs']
    colors = ['#FF6B6B' if i == class_names.index(result['predicted_class']) else '#4ECDC4'
              for i in range(len(class_names))]

    bars = ax2.barh(class_names, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()
    return fig


def create_demo_grid(results, save_path=None):

    n_images = len(results)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for idx, (ax, result) in enumerate(zip(axes, results)):
        ax.imshow(result['image'])
        ax.axis('off')
        title = f"{result['predicted_class']}\n({result['confidence']:.1%})"
        ax.set_title(title, fontsize=12, fontweight='bold')

    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved demo grid to {save_path}")

    plt.close()
    return fig


def run_demo(model_path: str, data_dir: str, results_dir: str, num_samples: int = 9):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)

    model, class_names = load_model(model_path, device)
    transform = get_transform()

    all_images = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                n = min(2, len(images))
                all_images.extend(random.sample(images, n))

    if len(all_images) > num_samples:
        all_images = random.sample(all_images, num_samples)

    print(f"\nProcessing {len(all_images)} sample images...")

    results = []
    for idx, img_path in enumerate(all_images):
        print(f"Processing image {idx+1}/{len(all_images)}: {os.path.basename(img_path)}")
        result = predict_image(model, img_path, transform, device, class_names)
        results.append(result)

        save_path = os.path.join(results_dir, f'prediction_{idx+1}.png')
        visualize_prediction(result, class_names, save_path)

    grid_path = os.path.join(results_dir, 'demo_grid.png')
    create_demo_grid(results, grid_path)

    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    for idx, result in enumerate(results, 1):
        print(f"{idx}. Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.2%})")

    print("\n" + "="*60)
    print(f"Results saved to: {results_dir}")
    print("="*60)

    return results

if __name__ == "__main__":

    MODEL_PATH = "../best_seresnet18.pth"
    DATA_DIR = "../Data"
    RESULTS_DIR = "../results"
    NUM_SAMPLES = 9

    random.seed(42)
    torch.manual_seed(42)

    try:
        results = run_demo(MODEL_PATH, DATA_DIR, RESULTS_DIR, NUM_SAMPLES)
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
