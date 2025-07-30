import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageFile
import numpy as np
import io

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Core Functions ---
@st.cache_resource
def get_pretrained_model():
    """Cached model loading to avoid reloading on every run"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    # Freeze all parameters
    for param in vgg.parameters():
        param.requires_grad_(False)

    # Define the layers to extract features from
    content_layers_idx = ['21']  # Layer for content (corresponds to conv4_2 in VGG19)
    style_layers_idx = ['0', '5', '10']  # Layers for style (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)

    # Create a new model that extracts features from specified layers
    model = ContentStyleModel(vgg, style_layers_idx, content_layers_idx).to(device).eval()
    return model, device

class ContentStyleModel(torch.nn.Module):
    def __init__(self, vgg_model, style_layers, content_layers):
        super(ContentStyleModel, self).__init__()
        self.vgg_model = vgg_model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.all_layers = sorted(list(set(style_layers + content_layers)), key=int)

    def forward(self, x):
        features = {}
        for name, layer in self.vgg_model._modules.items():
            x = layer(x)
            if name in self.all_layers:
                features[name] = x
        return features


def load_image(image_file, size=(256, 256)):
    """Load and preprocess image for VGG."""
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])
        return transform(image)
    except Exception as e:
        st.error(f"Image loading failed: {str(e)}")
        return None

def tensor_to_display(tensor):
    """Convert tensor to displayable image (undo normalization)."""
    image = tensor.clone().detach().cpu().squeeze(0)
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)

def gram_matrix(tensor):
    """Calculate the Gram matrix for style representation"""
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))  # Gram matrix
    return G.div(c * h * w)

def neural_style_transfer(content_img, style_img, model, num_steps=100, style_weight=1e6, content_weight=1):
    device = content_img.device
    target = content_img.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.01)
    content_layers = ['21']
    style_layers = ['0', '5', '10']

    content_features = model(content_img)
    style_features = model(style_img)

    for step in range(num_steps):
        optimizer.zero_grad()
        target_features = model(target)

        content_loss = content_weight * torch.nn.functional.mse_loss(
            target_features[content_layers[0]], content_features[content_layers[0]]
        )

        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = gram_matrix(style_features[layer])
            style_loss += torch.nn.functional.mse_loss(target_gram, style_gram)
        style_loss *= style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

    return target.detach()

# --- Application UI ---
st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and style image to blend them together")

col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Content Image", type=["jpg", "png", "jpeg"])
with col2:
    style_file = st.file_uploader("Style Image", type=["jpg", "png", "jpeg"])

# Display upload previews
if content_file and style_file:
    # Need to rewind the file pointers for displaying
    content_file_display = io.BytesIO(content_file.getvalue())
    style_file_display = io.BytesIO(style_file.getvalue())

    st.image([content_file_display, style_file_display],
             caption=["Content Image", "Style Image"],
             width=300)

# --- Processing ---
if st.button("✨ Apply Style Transfer"):
    if content_file and style_file:
        with st.spinner("Processing images... This may take a while for higher quality..."):
            try:
                # Load model and device
                model, device = get_pretrained_model()

                # Process images
                # IMPORTANT: Before passing to load_image, rewind the file pointers
                content_file.seek(0)
                style_file.seek(0)
                
                content_img = load_image(content_file).to(device)
                style_img = load_image(style_file).to(device)
                
                if content_img is None or style_img is None:
                    st.error("Failed to process images. Please check your image files.")
                    st.stop()
                
                # Perform style transfer
                output = neural_style_transfer(content_img, style_img, model)
                
                # Show results
                st.success("Complete!")
                st.image(tensor_to_display(output),
                         caption="Styled Result",
                         use_column_width=True)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.exception(e) # This will print the full traceback for debugging
    else:
        st.warning("Please upload both images first")