import io
import math
import os
import random
import zipfile

from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.models import efficientnet_b3


st.set_page_config(
    page_title="SAMD-IS GAN Generator + Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    h1 {
        color: #2C3E50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E8E8;
    }

    div.stButton > button {
        background-color: #27AE60;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #219150;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    [data-testid="stBaseButton-secondary"] {
        background-color: #ffffff;
        color: #555;
        border: 1px solid #ddd;
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================
# 1. MODEL ARCHITECTURE
# ==============================================================

class SelfAttention(nn.Module):
    def __init__(self, in_channels, max_tokens=4096):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.max_tokens = max_tokens

    def forward(self, x):
        batch, channels, height, width = x.shape
        stride = max(1, math.ceil(math.sqrt((height * width) / self.max_tokens)))
        x_small = F.avg_pool2d(x, kernel_size=stride, stride=stride) if stride > 1 else x

        b_small, _, h_small, w_small = x_small.shape
        q = self.q(x_small).view(b_small, -1, h_small * w_small).transpose(1, 2)
        k = self.k(x_small).view(b_small, -1, h_small * w_small)
        v = self.v(x_small).view(b_small, -1, h_small * w_small).transpose(1, 2)

        attn = torch.bmm(q, k) / math.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(b_small, channels, h_small, w_small)

        if stride > 1:
            out = F.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)
        return self.gamma * out + x


class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=6, img_size=128, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = img_size
        self.input_dim = latent_dim + n_classes
        self.init_size = img_size // 16

        self.l1 = nn.Linear(self.input_dim, 256 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(64, max_tokens=1024),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
        x = torch.cat([noise, one_hot], dim=1)
        x = self.l1(x)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    def __init__(self, img_size, n_classes, channels):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.n_classes = n_classes
        self.label_embedding = nn.Embedding(n_classes, 16)
        self.feature_extractor = None
        self.output_layer = None
        self._initialized = False

    def forward_features(self, x):
        if self.feature_extractor is None:
            raise NotImplementedError("Subclasses must define self.feature_extractor")
        return self.feature_extractor(x)

    def lock_feature_dim_and_initialize_fc(self):
        if self._initialized:
            return

        self.eval()
        current_device = next(self.parameters()).device
        with torch.no_grad():
            dummy_batch = 8
            dummy_img = torch.randn(dummy_batch, self.channels, self.img_size, self.img_size, device=current_device)
            dummy_label = torch.zeros(dummy_batch, dtype=torch.long, device=current_device)
            embedded = self.label_embedding(dummy_label).view(dummy_batch, 16, 1, 1)
            embedded = embedded.expand(-1, -1, self.img_size, self.img_size)
            x = torch.cat([dummy_img, embedded], dim=1)
            features = self.forward_features(x)
            feature_dim = features.reshape(dummy_batch, -1).size(1)

        self.output_layer = nn.Linear(feature_dim, 1).to(current_device)
        self._initialized = True
        self.train()

    def forward(self, img, labels):
        if not self._initialized:
            raise RuntimeError("Call lock_feature_dim_and_initialize_fc() before loading or inference.")

        batch_size, _, height, width = img.size()
        embedded = self.label_embedding(labels).view(batch_size, 16, 1, 1)
        embedded = embedded.expand(-1, -1, height, width)

        x = torch.cat([img, embedded], dim=1)
        x = self.forward_features(x)
        x = x.reshape(x.size(0), -1)
        return self.output_layer(x)


class TextureDiscriminator(Discriminator):
    def __init__(self, img_size, n_classes, channels):
        super().__init__(img_size, n_classes, channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels + 16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(128),

            SelfAttention(128, max_tokens=512),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(256),
        )


class StructureDiscriminator(Discriminator):
    def __init__(self, img_size, n_classes, channels):
        super().__init__(img_size, n_classes, channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels + 16, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
        )


class ColorDiscriminator(Discriminator):
    def __init__(self, img_size, n_classes, channels):
        super().__init__(img_size, n_classes, channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels + 16, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            nn.AdaptiveAvgPool2d(1),
        )


# ==============================================================
# 2. CONFIGURATION
# ==============================================================

GENERATOR_PATH = "G_epoch300.pth"
D_TEXTURE_PATH = "D_texture_epoch500.pth"
D_STRUCTURE_PATH = "D_structure_epoch500.pth"
D_COLOR_PATH = "D_color_epoch500.pth"
CLASSIFIER_PATH = "efficientnet_baseline_best.pth" 

LATENT_DIM = 100
NUM_CLASSES = 6
IMG_SIZE = 128
CHANNELS = 3
INTERNAL_BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Light Skin - Psoriasis",
    1: "Light Skin - Eczema",
    2: "Light Skin - Healthy",
    3: "Brown Skin - Psoriasis",
    4: "Brown Skin - Eczema",
    5: "Brown Skin - Healthy",
}

CLASSIFIER_PATH = "efficientnet_baseline_best.pth"
CLASSIFIER_NUM_CLASSES = 4

CLASSIFIER_CLASS_NAMES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
}


# ==============================================================
# 3. LOADING HELPERS
# ==============================================================

def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is neither a state_dict dict nor an nn.Module.")

    for key in ["model_state_dict", "state_dict", "model", "net", "generator", "G"]:
        if key in checkpoint:
            if isinstance(checkpoint[key], dict):
                return _strip_module_prefix(checkpoint[key])
            if isinstance(checkpoint[key], nn.Module):
                return checkpoint[key].state_dict()

    return _strip_module_prefix(checkpoint)


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


@st.cache_resource(show_spinner=False)
def load_generator_and_discriminators():
    messages = []

    if not os.path.exists(GENERATOR_PATH):
        raise FileNotFoundError(f"Generator weights not found: {GENERATOR_PATH}")

    generator = Generator(
        latent_dim=LATENT_DIM,
        n_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        channels=CHANNELS,
    ).to(DEVICE)

    generator_ckpt = _safe_torch_load(GENERATOR_PATH, DEVICE)
    generator_state = _extract_state_dict(generator_ckpt)
    generator.load_state_dict(generator_state, strict=False)
    generator.eval()

    disc_paths = {
        "texture": (TextureDiscriminator, D_TEXTURE_PATH),
        "structure": (StructureDiscriminator, D_STRUCTURE_PATH),
        "color": (ColorDiscriminator, D_COLOR_PATH),
    }

    discriminators = {}
    for name, (disc_cls, path) in disc_paths.items():
        if not os.path.exists(path):
            messages.append(f"{name.title()} discriminator not found: {path}")
            continue

        disc = disc_cls(IMG_SIZE, NUM_CLASSES, CHANNELS).to(DEVICE)
        disc.lock_feature_dim_and_initialize_fc()

        disc_ckpt = _safe_torch_load(path, DEVICE)
        disc_state = _extract_state_dict(disc_ckpt)
        disc.load_state_dict(disc_state, strict=False)
        disc.eval()
        discriminators[name] = disc

    if len(discriminators) != 3:
        missing = sorted(set(["texture", "structure", "color"]) - set(discriminators.keys()))
        if missing:
            messages.append("Falling back to heuristic ranking for missing: " + ", ".join(missing))

    return generator, discriminators, messages


@st.cache_resource(show_spinner=False)
def load_classifier(classifier_path):
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    checkpoint = _safe_torch_load(classifier_path, DEVICE)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(DEVICE)
        model.eval()
        return model

    state_dict = _extract_state_dict(checkpoint)

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        CLASSIFIER_NUM_CLASSES
    )
    model = model.to(DEVICE)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# ==============================================================
# 4. INFERENCE HELPERS
# ==============================================================

classifier_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def tensor_to_pil(img_tensor):
    img_tensor = (img_tensor + 1) / 2.0
    img_tensor = img_tensor.clamp(0, 1)
    return TF.to_pil_image(img_tensor).resize((256, 256), Image.NEAREST)


def convert_pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_zip_of_images(images, labels, seed_a, seed_b):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img_tensor in enumerate(images):
            img_pil = tensor_to_pil(img_tensor)
            file_name = f"Image_{i}_Class{labels}_Seed{seed_a}-{seed_b}.png"

            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zf.writestr(file_name, img_bytes.getvalue())

    return zip_buffer.getvalue()


def image_quality_score(img_tensor, label):
    img = (img_tensor + 1) / 2.0
    r, g, b = img[:, 0], img[:, 1], img[:, 2]

    grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
    edge_score = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
    spatial_std = torch.std(img, dim=[2, 3]).mean()

    mean_brightness = torch.mean(img)
    brightness_penalty = torch.abs(mean_brightness - 0.45)

    corner_tl = img[:, :, :8, :8].mean()
    corner_br = img[:, :, -8:, -8:].mean()
    black_border_penalty = torch.exp(-10 * (corner_tl + corner_br) / 2)

    if label in [0, 3]:
        redness = torch.mean(r) - torch.mean(g)
        contrast = torch.std(img, dim=[2, 3]).max()
        scale_texture = edge_score
        score = (
            2.0 * redness
            + 2.0 * scale_texture
            + 1.5 * contrast
            + 1.0 * spatial_std
            - 3.0 * brightness_penalty
            - 2.0 * black_border_penalty
        )
    elif label in [1, 4]:
        redness = torch.mean(r) - 0.5 * torch.mean(b)
        inflammation = torch.mean(r) - torch.mean(g)
        texture = edge_score
        score = (
            2.5 * redness
            + 2.0 * inflammation
            + 1.5 * texture
            + 1.0 * spatial_std
            - 3.0 * brightness_penalty
            - 2.0 * black_border_penalty
        )
    elif label in [2, 5]:
        smoothness = -edge_score
        evenness = -spatial_std
        score = (
            2.0 * smoothness
            + 1.5 * evenness
            - 3.0 * brightness_penalty
            - 2.0 * black_border_penalty
        )
    else:
        score = spatial_std + edge_score

    return score


@torch.no_grad()
def score_with_discriminators(images, labels, discriminators):
    texture_scores = discriminators["texture"](images, labels).view(-1)
    structure_scores = discriminators["structure"](images, labels).view(-1)
    color_scores = discriminators["color"](images, labels).view(-1)
    combined_scores = (texture_scores + structure_scores + color_scores) / 3.0

    return {
        "texture": texture_scores.detach().cpu(),
        "structure": structure_scores.detach().cpu(),
        "color": color_scores.detach().cpu(),
        "combined": combined_scores.detach().cpu(),
    }


@torch.no_grad()
def rank_generated_images(generated_imgs, labels, label_index, discriminators, ranking_method):
    if ranking_method == "Discriminator" and len(discriminators) == 3:
        disc_scores = score_with_discriminators(generated_imgs, labels, discriminators)
        ordered = torch.argsort(disc_scores["combined"], descending=True)
        return ordered.tolist(), disc_scores

    heuristic_scores = []
    for i in range(generated_imgs.shape[0]):
        heuristic_scores.append(image_quality_score(generated_imgs[i].unsqueeze(0), label_index).item())

    ordered = sorted(range(len(heuristic_scores)), key=lambda idx: heuristic_scores[idx], reverse=True)
    heuristic_tensor = torch.tensor(heuristic_scores)
    return ordered, {
        "combined": heuristic_tensor,
        "texture": heuristic_tensor,
        "structure": heuristic_tensor,
        "color": heuristic_tensor,
    }


@torch.no_grad()
def classify_pil_image(model, image_pil):
    image = image_pil.convert("RGB")
    x = classifier_transform(image).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())

    return pred_idx, probs.cpu()


# ==============================================================
# 5. APP UI
# ==============================================================

st.title("SAMD-IS CGAN Generator + Classifier")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D;'>Generate synthetic skin condition images, rank them with your discriminators, and classify uploaded images with EfficientNet-B3.</p>",
    unsafe_allow_html=True,
)
st.divider()

try:
    generator, discriminators, load_messages = load_generator_and_discriminators()
except Exception as exc:
    st.error(f"Failed to load generator/discriminators: {exc}")
    st.stop()

classifier_model = None
classifier_error = None

try:
    classifier_model = load_classifier(CLASSIFIER_PATH)
except Exception as exc:
    classifier_error = str(exc)

for message in load_messages:
    st.sidebar.warning(message)

if len(discriminators) == 3:
    st.sidebar.success("Generator + all 3 discriminators loaded")
else:
    st.sidebar.info("Generator loaded. Discriminator ranking is partially unavailable.")

if classifier_model is not None:
    st.sidebar.success("Classifier loaded")
else:
    st.sidebar.warning(f"Classifier unavailable: {classifier_error}")

if "seed_a" not in st.session_state:
    st.session_state.seed_a = 42
if "seed_b" not in st.session_state:
    st.session_state.seed_b = 100

st.sidebar.header("Settings")
label_index = st.sidebar.selectbox(
    "Select Condition",
    options=list(range(NUM_CLASSES)),
    index=0,
    format_func=lambda x: CLASS_NAMES.get(x, f"Class {x}"),
)
num_images = st.sidebar.slider("Number of Images", 1, 8, 4)
ranking_method = st.sidebar.radio(
    "Ranking Method",
    ["Discriminator", "Heuristic"],
    index=0 if len(discriminators) == 3 else 1,
    help="Discriminator uses your trained texture, structure, and color critics for cherry-picking.",
)
show_scores = st.sidebar.checkbox("Show ranking scores", value=True)

st.sidebar.divider()
st.sidebar.subheader("Morphing Controls")

if st.sidebar.button("Generate New Samples"):
    st.session_state.seed_a = random.randint(0, 10000)
    st.session_state.seed_b = random.randint(0, 10000)

col_seed1, col_seed2 = st.sidebar.columns(2)
with col_seed1:
    seed_a = int(st.number_input("Start Seed", key="seed_a", step=1))
with col_seed2:
    seed_b = int(st.number_input("Target Seed", key="seed_b", step=1))

alpha = st.sidebar.slider(
    "Morph Factor",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Slide to morph between the Start and Target seeds.",
)

# ==============================================================
# 6. GENERATION
# ==============================================================

torch.manual_seed(seed_a)
noise_a = torch.randn(INTERNAL_BATCH, LATENT_DIM, device=DEVICE)

torch.manual_seed(seed_b)
noise_b = torch.randn(INTERNAL_BATCH, LATENT_DIM, device=DEVICE)

noise_interp = (1 - alpha) * noise_a + alpha * noise_b
labels = torch.full((INTERNAL_BATCH,), label_index, dtype=torch.long, device=DEVICE)

with torch.no_grad():
    generated_imgs = generator(noise_interp, labels)

ordered_indices, score_dict = rank_generated_images(
    generated_imgs=generated_imgs,
    labels=labels,
    label_index=label_index,
    discriminators=discriminators,
    ranking_method=ranking_method,
)

best_indices = ordered_indices[:num_images]
generated_imgs = generated_imgs[best_indices]
noise_a = noise_a[best_indices]
noise_b = noise_b[best_indices]
noise_interp = noise_interp[best_indices]

selected_scores = {
    key: value[best_indices].tolist() if hasattr(value, "tolist") else value
    for key, value in score_dict.items()
}

# ==============================================================
# 7. DISPLAY
# ==============================================================

st.markdown(
    f"### Results: <span style='color:#2E86C1'>{CLASS_NAMES.get(label_index, f'Class {label_index}')}</span>",
    unsafe_allow_html=True,
)
st.caption(f"Ranking used: **{ranking_method}**")

if alpha == 0.0:
    st.caption(f"Showing pure **Start Seed ({seed_a})**")
elif alpha == 1.0:
    st.caption(f"Showing pure **Target Seed ({seed_b})**")
else:
    st.caption(f"Morphing: **{int(alpha * 100)}%** transition from Seed {seed_a} to {seed_b}")

tab_grid, tab_single, tab_classifier = st.tabs(["Grid View", "Single Focus", "Classifier"])

with tab_grid:
    zip_bytes = create_zip_of_images(generated_imgs, label_index, seed_a, seed_b)
    st.download_button(
        label="Download All Images (ZIP)",
        data=zip_bytes,
        file_name=f"Batch_Class{label_index}_Seed{seed_a}-{seed_b}.zip",
        mime="application/zip",
    )

    cols = st.columns(4)
    for i, img_tensor in enumerate(generated_imgs):
        img_pil = tensor_to_pil(img_tensor)
        with cols[i % 4]:
            st.image(img_pil, use_container_width=True)
            if show_scores:
                st.caption(f"Score: {selected_scores['combined'][i]:.4f}")
                if ranking_method == "Discriminator" and len(discriminators) == 3:
                    st.caption(
                        " | ".join(
                            [
                                f"T {selected_scores['texture'][i]:.3f}",
                                f"S {selected_scores['structure'][i]:.3f}",
                                f"C {selected_scores['color'][i]:.3f}",
                            ]
                        )
                    )

with tab_single:
    col_select, col_display = st.columns([1, 3])

    with col_select:
        st.info("Select an image from the batch to inspect its specific morphing path.")
        selected_idx = st.selectbox("Choose Image Number", range(num_images))
        if show_scores:
            st.metric("Selected score", f"{selected_scores['combined'][selected_idx]:.4f}")
            if ranking_method == "Discriminator" and len(discriminators) == 3:
                st.write(
                    {
                        "texture": round(selected_scores["texture"][selected_idx], 4),
                        "structure": round(selected_scores["structure"][selected_idx], 4),
                        "color": round(selected_scores["color"][selected_idx], 4),
                    }
                )

    with col_display:
        n_a = noise_a[selected_idx].unsqueeze(0)
        n_b = noise_b[selected_idx].unsqueeze(0)
        n_curr = noise_interp[selected_idx].unsqueeze(0)
        l_single = torch.full((1,), label_index, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            img_start = generator(n_a, l_single)
            img_target = generator(n_b, l_single)
            img_current = generator(n_curr, l_single)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Start (Seed A)")
            st.image(tensor_to_pil(img_start[0]), use_container_width=True)

        with c2:
            st.caption("Current Morph")
            main_pil = tensor_to_pil(img_current[0])
            st.image(main_pil, use_container_width=True)
            st.download_button(
                label="Download Current Morph",
                data=convert_pil_to_bytes(main_pil),
                file_name=f"Focus_Img_{selected_idx}_Morph{int(alpha * 100)}.png",
                mime="image/png",
                key=f"dl_focus_{selected_idx}",
            )

        with c3:
            st.caption("Target (Seed B)")
            st.image(tensor_to_pil(img_target[0]), use_container_width=True)

with tab_classifier:
    st.subheader("Skin Condition Classifier")

    if classifier_model is None:
        st.error(f"Classifier failed to load: {classifier_error}")
    else:
        uploaded_file = st.file_uploader(
            "Upload a skin image",
            type=["jpg", "jpeg", "png"],
            key="classifier_upload",
        )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        with col_pred:
            pred_idx, probs = classify_pil_image(classifier_model, uploaded_image)

            st.success(
                f"Prediction: {CLASSIFIER_CLASS_NAMES.get(pred_idx, f'Class {pred_idx}')}"
            )
            st.write(f"Confidence: {probs[pred_idx].item() * 100:.2f}%")

            top_k = min(3, CLASSIFIER_NUM_CLASSES)
            top_probs, top_idxs = torch.topk(probs, k=top_k)

            st.markdown("**Top predictions**")
            for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_idxs.tolist()), start=1):
                label_name = CLASSIFIER_CLASS_NAMES.get(idx, f"Class {idx}")
                st.write(f"{rank}. {label_name} — {prob * 100:.2f}%")

            st.markdown("**All class probabilities**")
            for idx in range(CLASSIFIER_NUM_CLASSES):
                label_name = CLASSIFIER_CLASS_NAMES.get(idx, f"Class {idx}")
                st.caption(f"{label_name}: {probs[idx].item() * 100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("**Created by:**")
st.sidebar.caption("Maria Rafaela Pelagio \n Sophia Danielle Salta\n Veneza Vielle Vergara")