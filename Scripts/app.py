import os, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import librosa
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))
#login(token=os.environ.get("----------"))
REPO_ID  = "xytan2022/checkpoints"   
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP  = False   # set False for CPU inference on HF Spaces

GENRES   = ['blues','classical','country','disco','hiphop',
            'jazz','metal','pop','reggae','rock']
id2label = {i: g for i, g in enumerate(GENRES)}
NUM_LABELS = len(GENRES)

AST_MODEL_NAME  = 'MIT/ast-finetuned-audioset-10-10-0.4593'
AST_SAMPLE_RATE = 16000
AST_DURATION    = 20
AST_MAX_LENGTH  = AST_SAMPLE_RATE * AST_DURATION
AST_N_TTA       = 5

CNN_SAMPLE_RATE = 22050
CNN_DURATION    = 5.0
CNN_N_SAMPLES   = int(CNN_SAMPLE_RATE * CNN_DURATION)
CNN_N_FFT       = 2048
CNN_HOP_LEN     = 512
CNN_N_MELS      = 128

ENSEMBLE_WEIGHTS = {
    'AST': 0.40, 'CustomCNN': 0.15,
    'AudioResNet34': 0.20, 'EfficientNetB0': 0.25,
}
CKPT_FILES = {
    'AST'           : 'messy-mashup_checkpoints_ast_best_phase2.pth',
    'CustomCNN'     : 'CustomCNN_best.pth',
    'AudioResNet34' : 'messy-mashup_checkpoints_checkpoints_AudioResNet34_best.pth',
    'EfficientNetB0': 'EfficientNetB0_best.pth',
}

# ─── Model Definitions ────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_LABELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))


class AudioResNet(nn.Module):
    def __init__(self, num_classes=NUM_LABELS):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        orig_w = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight[:,0] = orig_w.mean(dim=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x): return self.resnet(x)


class AudioEfficientNet(nn.Module):
    def __init__(self, num_classes=NUM_LABELS):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        orig_w = self.model.features[0][0].weight.clone()
        self.model.features[0][0] = nn.Conv2d(1,32,3,stride=2,padding=1,bias=False)
        with torch.no_grad():
            self.model.features[0][0].weight[:,0] = orig_w.mean(dim=1)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes)
    def forward(self, x): return self.model(x)


# ─── GPU/CPU Mel Transform ────────────────────────────────
gpu_mel_transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=CNN_SAMPLE_RATE, n_fft=CNN_N_FFT,
                     hop_length=CNN_HOP_LEN, n_mels=CNN_N_MELS),
    T.AmplitudeToDB()
).to(DEVICE)


# ─── Load Checkpoints (runs once at startup) ──────────────
print("Loading checkpoints...")

def _dl(name):
    return hf_hub_download(REPO_ID, CKPT_FILES[name], repo_type="dataset")

def _load(model, name):
    ckpt = torch.load(_dl(name), map_location=DEVICE)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(DEVICE)

# AST
ast_feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
_ast_base = ASTForAudioClassification.from_pretrained(
    AST_MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True).to(DEVICE)
ast_state = torch.load(_dl('AST'), map_location=DEVICE)
ast_state = {k.replace('module.', ''): v for k, v in ast_state.items()}
_ast_base.load_state_dict(ast_state, strict=True)
_ast_base.eval()
print("[OK] AST loaded")

cnn_model1 = _load(CustomCNN(),      'CustomCNN');     print("[OK] CustomCNN loaded")
cnn_model2 = _load(AudioResNet(),    'AudioResNet34'); print("[OK] AudioResNet34 loaded")
cnn_model3 = _load(AudioEfficientNet(), 'EfficientNetB0'); print("[OK] EfficientNetB0 loaded")


# ─── Audio Helpers ────────────────────────────────────────
def ast_load_audio(path):
    audio, _ = librosa.load(path, sr=AST_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)

def ast_normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def ast_crop_random(audio):
    if len(audio) >= AST_MAX_LENGTH:
        start = random.randint(0, len(audio) - AST_MAX_LENGTH)
        return audio[start : start + AST_MAX_LENGTH]
    return np.pad(audio, (0, AST_MAX_LENGTH - len(audio)))


# ─── Inference Helpers ────────────────────────────────────
def ast_predict_proba(audio):
    probs = []
    for _ in range(AST_N_TTA):
        c   = ast_normalize(ast_crop_random(audio))
        inp = ast_feature_extractor(c, sampling_rate=AST_SAMPLE_RATE, return_tensors='pt')
        with torch.no_grad():
            out = _ast_base(input_values=inp['input_values'].to(DEVICE))
            probs.append(torch.softmax(out.logits, 1).cpu().numpy()[0])
    return np.mean(probs, axis=0)


def cnn_predict_proba(audio_path):
    try:
        wav, sr = torchaudio.load(audio_path)
        if sr != CNN_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, CNN_SAMPLE_RATE)
        wav = wav.mean(0).numpy()
    except:
        wav = np.zeros(CNN_N_SAMPLES)

    L, n = len(wav), CNN_N_SAMPLES
    if L < n:
        wav = np.pad(wav, (0, n - L)); L = n

    chunks = [
        wav[:n],
        wav[max(0, L//2 - n//2) : max(0, L//2 - n//2) + n],
        wav[L - n:]
    ]
    chunk_tensors = torch.stack([
        torch.tensor(
            c[:n] if len(c) >= n else np.pad(c, (0, n - len(c))),
            dtype=torch.float32
        ) for c in chunks
    ]).unsqueeze(1).to(DEVICE)

    specs = gpu_mel_transform(chunk_tensors.squeeze(1).unsqueeze(1))

    probs = []
    with torch.no_grad():
        for m in [cnn_model1, cnn_model2, cnn_model3]:
            out = m(specs)
            probs.append(torch.softmax(out, 1).cpu().numpy())

    return np.mean(probs[0], axis=0), np.mean(probs[1], axis=0), np.mean(probs[2], axis=0)


# ─── Main Predict Function ────────────────────────────────
def predict(audio_path):
    if audio_path is None:
        return "No audio uploaded", {}

    p_ast          = ast_predict_proba(ast_load_audio(audio_path))
    p_c1, p_c2, p_c3 = cnn_predict_proba(audio_path)

    ep = (ENSEMBLE_WEIGHTS['AST']            * p_ast +
          ENSEMBLE_WEIGHTS['CustomCNN']       * p_c1  +
          ENSEMBLE_WEIGHTS['AudioResNet34']   * p_c2  +
          ENSEMBLE_WEIGHTS['EfficientNetB0']  * p_c3)

    top_genre = id2label[int(np.argmax(ep))]
    scores    = {GENRES[i]: float(round(ep[i], 4)) for i in range(NUM_LABELS)}
    sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
    return top_genre, sorted_scores


# ─── Gradio UI ────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload audio (.wav or .mp3)"),
    outputs=[
        gr.Text(label="Predicted Genre"),
        gr.Label(num_top_classes=10, label="Confidence Scores")
    ],
    title="🎵 Messy Mashup — Music Genre Classifier",
    description=(
        "Ensemble of **AST (40%) + EfficientNetB0 (25%) + "
        "AudioResNet34 (20%) + CustomCNN (15%)**\n\n"
        "Upload a song clip to classify it into one of 10 genres: "
        "Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock."
    ),
    examples=[]
)

demo.launch()
