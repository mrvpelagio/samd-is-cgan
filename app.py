import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms.functional as TF
import numpy as np
import random
import io
import zipfile
from PIL import Image


st.set_page_config(
    page_title="SAMD-IS GAN Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Title Styling */
    h1 {
        color: #2C3E50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E8E8;
    }

    /* Customizing the Generate Button */
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

    /* Secondary/Download Button Styling */
    [data-testid="stBaseButton-secondary"] {
        background-color: #ffffff;
        color: #555;
        border: 1px solid #ddd;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

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
        B, C, H, W = x.shape
        s = max(1, math.ceil(math.sqrt((H*W)/self.max_tokens)))
        if s > 1:
            x_small = F.avg_pool2d(x, kernel_size=s, stride=s)
        else:
            x_small = x

        Bh, Ch, Hs, Ws = x_small.shape
        q = self.q(x_small).view(Bh, -1, Hs*Ws).transpose(1, 2)
        k = self.k(x_small).view(Bh, -1, Hs*Ws)
        v = self.v(x_small).view(Bh, -1, Hs*Ws).transpose(1, 2)

        attn = torch.bmm(q, k) / math.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(Bh, C, Hs, Ws)

        if s > 1:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
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

# ==============================================================
# 2. APP CONFIGURATION
# ==============================================================

MODEL_PATH = "g_ema.pth"  
LATENT_DIM = 100
NUM_CLASSES = 6 
IMG_SIZE = 128
device = torch.device("cpu")

@st.cache_resource
def load_model():
    try:
        model = Generator(latent_dim=LATENT_DIM, n_classes=NUM_CLASSES, img_size=IMG_SIZE)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

def tensor_to_pil(img_tensor):
    img_tensor = (img_tensor + 1) / 2.0
    img_pil = TF.to_pil_image(img_tensor).resize((256, 256), Image.NEAREST)
    return img_pil

def convert_pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_zip_of_images(images, labels, seed_a, seed_b):
    """
    Compresses all images in the batch into a single ZIP file in memory.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img_tensor in enumerate(images):
            # Convert to PIL
            img_pil = tensor_to_pil(img_tensor)
            
            # Create a filename
            file_name = f"Image_{i}_Class{labels}_Seed{seed_a}-{seed_b}.png"
            
            # Convert PIL to bytes
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            
            # Write to zip
            zf.writestr(file_name, img_bytes.getvalue())
            
    return zip_buffer.getvalue()

def image_quality_score(img_tensor, label):
    variance_score = torch.var(img_tensor)

    grad_x = img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]
    grad_y = img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]
    edge_score = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

    mean_brightness = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

    if label in [2,5]:
        smoothness = -edge_score
        brightness_penalty = -torch.abs(mean_brightness)
        return smoothness + brightness_penalty
    else: 
        return variance_score + edge_score
    
def color_std(img_tensor):
    # img tensor shape: (1, 3, H, W)
    
    spatial_std = torch.std(img_tensor, dim=[2,3])
    mean_std = torch.mean(spatial_std)

    return mean_std

def healthy_score(img_tensor):
    spatial_std = torch.std(img_tensor, dim=[2,3]).mean()
    brightness = torch.mean(img_tensor)
    brightness_penalty =torch.abs(brightness)

    return spatial_std + 0.3 * brightness_penalty


# ui
st.title("SAMD-IS CGAN Generator")
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Generate synthetic skin condition images and <b>morph</b> between them in real-time.</p>", unsafe_allow_html=True)
st.divider()

# Initialize model
model = load_model()

# --- Session State Initialization ---
if 'seed_a' not in st.session_state:
    st.session_state.seed_a = 42
if 'seed_b' not in st.session_state:
    st.session_state.seed_b = 100

if model:
    # --- Sidebar Controls ---
    st.sidebar.header("Settings")
    
    # 1. Class Selection
    class_names = {
        0: "Light Skin - Psoriasis",
        1: "Light Skin - Eczema",
        2: "Light Skin - Healthy",
        3: "Brown Skin - Psoriasis",
        4: "Brown Skin - Eczema",
        5: "Brown Skin - Healthy"
    }

    label_index = st.sidebar.selectbox(
        "Select Condition", 
        options=list(range(NUM_CLASSES)),
        index=0,
        format_func=lambda x: class_names.get(x, f"Class {x}")
    )
    
    # 2. Batch Size
    num_images = st.sidebar.slider("Number of Images", 1, 8, 4)
    
    st.sidebar.divider()
    st.sidebar.subheader("Morphing Controls")
    
    # 3. The Generate Button
    if st.sidebar.button("Generate New Samples"):
        st.session_state.seed_a = random.randint(0, 10000)
        st.session_state.seed_b = random.randint(0, 10000)
    
    # 4. Seeds
    col_seed1, col_seed2 = st.sidebar.columns(2)
    with col_seed1:
        seed_a = st.number_input("Start Seed", key='seed_a')
    with col_seed2:
        seed_b = st.number_input("Target Seed", key='seed_b')
        
    # 5. The Morph Slider
    alpha = st.sidebar.slider(
        "Morph Factor", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.05,
        help="Slide to morph between the Start and Target seeds. This is a new version of the app."
    )

    # --- Main Generation Logic ---
    internal_batch = 32

    normal_labels = [2, 5] # healthy skin won't be applied cherry picking 
    # Generate Start Noise (Batch A)
    torch.manual_seed(seed_a)
    noise_a = torch.randn(internal_batch, LATENT_DIM, device=device)
    
    # Generate Target Noise (Batch B)
    torch.manual_seed(seed_b)
    noise_b = torch.randn(internal_batch, LATENT_DIM, device=device)
    
    # Interpolate
    noise_interp = (1 - alpha) * noise_a + alpha * noise_b
    
    # Create Labels
    labels = torch.full((internal_batch,), label_index, dtype=torch.long, device=device)
    
    # Run Model for Current State
    with torch.no_grad():
        generated_imgs = model(noise_interp, labels)

#    scores = []

#    for i in range(generated_imgs.shape[0]):
#        score = image_quality_score(generated_imgs[i].unsqueeze(0))
#        scores.append((score.item(), i))

#    scores.sort(reverse=True)
#    best_indices = [idx for _, idx in scores[:num_images]]
#    generated_imgs = generated_imgs[best_indices]

if label_index not in normal_labels: # apply cherry picking
    scores = []

    for i in range(generated_imgs.shape[0]):
        score = image_quality_score(generated_imgs[i].unsqueeze(0), label_index)
        scores.append((score.item(), i))

    scores.sort(reverse=True)
    best_indices = [idx for _, idx in scores[:num_images]]
    generated_imgs = generated_imgs[best_indices]
    noise_a = noise_a[best_indices]
    noise_b = noise_b[best_indices]
    noise_interp = noise_interp[best_indices]

else: # healthy 
    scores = []

    for i in range(generated_imgs.shape[0]):
        img = generated_imgs[i].unsqueeze(0)
        score = healthy_score(img)
        scores.append((score.item(), i))

    scores.sort()
    best_indices = [idx for _, idx in scores[:num_images]]

    generated_imgs = generated_imgs[best_indices]
    noise_a = noise_a[best_indices]
    noise_b = noise_b[best_indices]
    noise_interp = noise_interp[best_indices]


    # --- Display Results ---
st.markdown(f"### Results: <span style='color:#2E86C1'>{class_names.get(label_index, f'Class {label_index}')}</span>", unsafe_allow_html=True)
    
if alpha == 0.0:
    st.caption(f"Showing pure **Start Seed ({seed_a})**")
elif alpha == 1.0:
    st.caption(f"Showing pure **Target Seed ({seed_b})**")
else:
    st.caption(f"Morphing: **{int(alpha*100)}%** transition from Seed {seed_a} to {seed_b}")

    # --- TABS FOR VIEW MODES ---
tab_grid, tab_single = st.tabs(["Grid View", "Single Focus"])

    # --- TAB 1: GRID VIEW ---
with tab_grid:
        # Add Download All Button at the top
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

    # --- TAB 2: SINGLE FOCUS ---
with tab_single:
    col_select, col_display = st.columns([1, 3])
        
    with col_select:
        st.info("Select an image from the batch to inspect its specific morphing path.")
        selected_idx = st.selectbox("Choose Image Number", range(num_images))
    
    with col_display:
        # We need to generate the Start and Target versions just for this specific index
        # to show the comparison
            
            # Get specific noise vectors for this index
        n_a = noise_a[selected_idx].unsqueeze(0)
        n_b = noise_b[selected_idx].unsqueeze(0)
        n_curr = noise_interp[selected_idx].unsqueeze(0)
        l_single = torch.full((1,), label_index, dtype=torch.long, device=device)
            
        with torch.no_grad():
            img_start = model(n_a, l_single)
            img_target = model(n_b, l_single)
            img_current = model(n_curr, l_single)

            # Display Comparison
            c1, c2, c3 = st.columns(3)
            
        with c1:
            st.caption("Start (Seed A)")
            st.image(tensor_to_pil(img_start[0]), use_container_width=True)
            
        with c2:
            st.caption("Current Morph")
            # Make the main one larger
            main_pil = tensor_to_pil(img_current[0])
            st.image(main_pil, use_container_width=True)
                
                # Download for main image
            st.download_button(
                label="Download Current Morph",
                data=convert_pil_to_bytes(main_pil),
                file_name=f"Focus_Img_{selected_idx}_Morph{int(alpha*100)}.png",
                mime="image/png",
                key=f"dl_focus_{selected_idx}"
            )
                
        with c3:
            st.caption("Target (Seed B)")
            st.image(tensor_to_pil(img_target[0]), use_container_width=True)




# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by:**")
st.sidebar.caption("Maria Rafaela Pelagio \n         Sophia Danielle Salta\n Veneza Vielle Vergara")