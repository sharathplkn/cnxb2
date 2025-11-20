import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import os
import io

# --- Configuration ---
MODEL_PATH = "best_model_MIMIC_FINAL.pth"
TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Model Config (must match the trained model)
D_MODEL = 768
ENCODER_DIM = 768
NHEAD = 8
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
IMAGE_SIZE = 512
MAX_LENGTH = 100

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL CLASS DEFINITIONS (Untouched) ---
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = CustomLayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                                                 requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x

class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            CustomLayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                CustomLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x

class LanguageDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, encoder_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_projection = nn.Linear(encoder_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder_projection(memory)
        output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ImageToTextModel(nn.Module):
    def __init__(self, encoder, decoder, pad_token_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id

    def forward(self, images, tgt_tokens):
        img_features = self.encoder(images)
        img_features = img_features.unsqueeze(1)
        tgt_input = tgt_tokens[:, :-1]
        tgt_expected = tgt_tokens[:, 1:]
        logits = self.decoder(tgt=tgt_input, memory=img_features)
        return logits, tgt_expected

    def generate(self, image, tokenizer, max_len=256):
        self.eval()
        with torch.no_grad():
            img_features = self.encoder(image).unsqueeze(1)
            output_tokens = torch.tensor([tokenizer.cls_token_id], dtype=torch.long).unsqueeze(0).to(image.device)

            for _ in range(max_len - 1):
                logits = self.decoder(tgt=output_tokens, memory=img_features)
                last_logits = logits[:, -1, :]
                _, next_token = torch.max(last_logits, dim=-1)
                output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == tokenizer.sep_token_id:
                    break
         
        return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# --- Streamlit Caching Functions (Untouched) ---
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(TOKENIZER_NAME)

@st.cache_resource
def load_model(model_path):
    tokenizer = load_tokenizer()
    PAD_TOKEN_ID = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size

    encoder = ConvNeXtEncoder(in_chans=3, dims=[96, 192, 384, ENCODER_DIM]).to(device)
    decoder = LanguageDecoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        encoder_dim=ENCODER_DIM
    ).to(device)
    model = ImageToTextModel(encoder, decoder, PAD_TOKEN_ID).to(device)

    if not os.path.exists(model_path):
        st.error(f"Model file not found: '{model_path}'. Please make sure 'best_model_MIMIC_FINAL.pth' is in the same directory as this app.")
        return None
     
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# --- Image Transform (Untouched) ---
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Main App ---
st.set_page_config(
    page_title="MedVision AI - Chest X-Ray Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Header
st.title("🩺 MedVision AI")
st.markdown("## Clinical-Grade Chest X-Ray Analysis & Diagnostic Reporting")

# Load model and tokenizer
with st.spinner("Initializing system..."):
    tokenizer = load_tokenizer()
    model = load_model(MODEL_PATH)

if model is None:
    st.error("❌ System initialization failed. Please verify model configuration.")
    st.stop()

# Simple Sidebar
with st.sidebar:
    st.header("📤 Medical Imaging")
    st.info("Upload chest X-ray for AI analysis")
     
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        help="Select a chest X-ray image for diagnostic analysis"
    )
     
    if uploaded_file is not None:
        st.success("✅ Medical Image Ready for Analysis")
         
        st.subheader("File Information")
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.1f} KB",
            "File Type": uploaded_file.type.split('/')[-1].upper(),
            "Dimensions": "To be analyzed"
        }
         
        st.markdown(
            "\n".join(f"* **{key}**: {value}" for key, value in file_details.items())
        )
         
    st.markdown("---")
    st.subheader("Clinical System Info")
    st.markdown("* **Version**: v2.1")
    st.markdown("* **Status**: Operational")


# Main Content
if uploaded_file is not None:
    # Read and process image
    image_data = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
     
    # Session state management
    if 'generated_report' not in st.session_state:
        st.session_state['generated_report'] = ""
     
    # Two-column simple layout
    col1, col2 = st.columns(2)
     
    with col1:
        st.header("🖼️ Medical Imaging")
        st.image(pil_image, caption=uploaded_file.name, use_column_width=True)
        st.caption("High-resolution digital chest radiograph")
     
    with col2:
        st.header("📋 Diagnostic Report")
         
        # Simple Generate Report Button
        if st.button("🔍 **Generate Comprehensive Analysis**", use_container_width=True):
            with st.spinner("🔄 Advanced AI analysis in progress..."):
                try:
                    # Generate report
                    report_text = model.generate(
                        image=inference_transform(pil_image).unsqueeze(0).to(device),
                        tokenizer=tokenizer,
                        max_len=MAX_LENGTH
                    )
                     
                    # Store in session state
                    st.session_state['generated_report'] = report_text
                    st.rerun()
                     
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
     
        # Simple Report Display
        if st.session_state['generated_report']:
            st.success("✅ Comprehensive Analysis Complete")
             
            st.subheader("📄 CLINICAL FINDINGS")
            st.text_area(
                "AI-Generated Report",
                st.session_state['generated_report'],
                height=300
            )
             
            st.markdown("---")
            
            # New Analysis Button
            if st.button("🔄 New Analysis", use_container_width=False):
                st.session_state['generated_report'] = ""
                st.rerun()
     
        else:
            st.info("📋 Diagnostic Report Pending. Click the button to start analysis.")
            st.markdown("""
            * **Instructions**: Initiate AI analysis to generate comprehensive diagnostic findings.
            * **Disclaimer**: Clinical decision support tool - Always verify with radiologist.
            """)

else:
    # Simple Welcome Screen
    st.info("Upload a chest X-ray image in the sidebar to begin analysis.")
    st.markdown("---")
    st.header("Welcome to MedVision AI")
    st.markdown("""
        Advanced artificial intelligence system for **comprehensive chest X-ray analysis** and automated diagnostic reporting.
    """)
    st.markdown("---")
    st.subheader("Clinical Workflow")
    st.markdown("""
    1.  **📤 Upload**: Select chest X-ray image via sidebar.
    2.  **🔍 Analyze**: Click the 'Generate' button to process the image with clinical algorithms.
    3.  **📋 Review**: The comprehensive diagnostic report will be generated for your review.
    """)
    st.warning("""
        **⚕️ Clinical Note**: This AI system assists healthcare professionals and should 
        be used in conjunction with radiologist interpretation and clinical judgment.
    """)

# Simple Footer
st.markdown("---")
st.markdown("<footer><p style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>🩺 MedVision AI • Advanced Medical Imaging Analysis Platform</p></footer>", unsafe_allow_html=True)
