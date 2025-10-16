import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import traceback
import json

# Simple Gradio app that tries to load a ResNet50 from a .pth file in the same folder.
# It supports both full-model checkpoints and state_dicts.

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# choose device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_model_file(max_up_levels=4):
    """Search for .pth/.pt files in current folder and up to `max_up_levels` parent directories.
    Returns the first preferred filename found, otherwise the first .pth/.pt discovered.
    """
    # prefer the weights/state_dict file first so users can supply a .pth containing only weights
    preferred = [
        'food101_resnet50_weights.pth',
        'food101_resnet50_full.pth',
        'food101_resnet50_state_dict.pth',
    ]

    # walk up parent directories
    cur = MODEL_DIR
    candidates = []
    for _ in range(max_up_levels + 1):
        try:
            entries = os.listdir(cur)
        except Exception:
            entries = []

        # look for preferred names first
        for p in preferred:
            if p in entries:
                return os.path.join(cur, p)

        # collect any .pth/.pt files
        for f in entries:
            if f.endswith('.pth') or f.endswith('.pt'):
                candidates.append(os.path.join(cur, f))

        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent

    if candidates:
        return candidates[0]
    return None


def load_checkpoint(path, device='cpu'):
    """Compact loader: accepts pickled nn.Module or state_dict-like checkpoints.
    Returns an nn.Module on success or raises on failure.
    """
    print(f"Loading model from: {path}")
    basename = os.path.basename(path).lower()

    # allowlist removed per user request; we no longer register ResNet globals for safe unpickling

    # choose load order: prefer safer weights_only unless filename indicates a full pickle
    order = [False, True] if 'full' in basename else [True, False]
    data = None
    for weights_only in order:
        try:
            data = torch.load(path, map_location=device, weights_only=weights_only)
            break
        except TypeError:
            # older torch may not accept weights_only kwarg
            try:
                data = torch.load(path, map_location=device)
                break
            except Exception:
                continue
        except Exception:
            continue
    if data is None:
        raise RuntimeError('Failed to load checkpoint')

    if isinstance(data, nn.Module):
        return data.eval()

    if isinstance(data, dict):
        # prefer embedded module
        for v in data.values():
            if isinstance(v, nn.Module):
                return v.eval()

        # common keys
        state_dict = data.get('state_dict') or data.get('model_state_dict') or data.get('model')
        if state_dict is None:
            try:
                if all(hasattr(v, 'dim') for v in data.values()):
                    state_dict = data
            except Exception:
                state_dict = None

        if isinstance(state_dict, dict):
            # build ResNet50 for Food-101 and load weights
            try:
                try:
                    model = models.resnet50(weights=None, num_classes=101)
                except TypeError:
                    model = models.resnet50(pretrained=False, num_classes=101)

                new_state = { (k[len('module.'):] if k.startswith('module.') else k): v
                              for k, v in state_dict.items() }
                model.load_state_dict(new_state, strict=False)
                return model.eval()
            except Exception as e:
                raise RuntimeError('Failed to load state_dict into ResNet50') from e

    raise RuntimeError('Unsupported checkpoint format')


# image preprocessing (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = None
print('MODEL_DIR =', MODEL_DIR)
try:
    print('MODEL_DIR contents:', sorted(os.listdir(MODEL_DIR)))
except Exception as e:
    print('Error listing MODEL_DIR:', e)

model_path = find_model_file()
print('find_model_file() ->', model_path)
if model_path:
    try:
        model = load_checkpoint(model_path, device=device)
        # ensure model is on the correct device
        try:
            model = model.to(device)
        except Exception:
            # some pickled models may already be on correct device or not support to(device)
            pass
        print('Model loaded successfully')
    except Exception as e:
        print('Error loading model:', e)
        traceback.print_exc()
else:
    print('No .pth/.pt model file found in the folder; the app will still run but return a helpful message.')
    # imports are at top-level; nothing else to do here

# load labels mapping if available
LABELS = None
labels_path = os.path.join(MODEL_DIR, 'food101_labels.json')
if os.path.exists(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            # expect structure {"id2label": {"0": "apple_pie", ...}}
            if isinstance(data, dict) and 'id2label' in data:
                LABELS = data['id2label']
                print('Loaded labels mapping with', len(LABELS), 'entries')
            else:
                print('labels file found but unexpected format')
    except Exception as e:
        print('Failed to load labels file:', e)
else:
    print('No labels file found at', labels_path)


def predict(image: Image.Image):
    """Return top-5 class indices and probabilities. If no model, return message."""
    if model is None:
        return 'No model loaded. Place a .pth or .pt file in the app folder and restart the container.'

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    x = transform(image).unsqueeze(0)
    # move input to model device
    try:
        x = x.to(device)
    except Exception:
        pass
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        topk = torch.topk(probs, k=5)
        results = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            # map index to label if LABELS is available (labels keys are strings)
            if LABELS is not None:
                label = LABELS.get(str(idx), f'class_{idx}')
            else:
                label = f'class_{idx}'
            results.append((label, round(float(score), 4)))
        return {label: score for label, score in results}


title = 'Food101 ResNet50 — Gradio demo'
description = 'Upload an image and the app will return top-5 predictions (indices) using the first .pth file found in this folder.'

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    title=title,
    description=description,
)


if __name__ == '__main__':
    # listen on all interfaces so the container can expose the service
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
