import os
import json
import torch
from torchvision import models, transforms
import gradio as gr

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
try:
    model = models.resnet50(weights=None, num_classes=101)
except TypeError:
    model = models.resnet50(pretrained=False, num_classes=101)

state_dict = torch.load('food101_resnet50_weights.pth', map_location=device, weights_only=True)
new_state = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
model.load_state_dict(new_state, strict=False)
model = model.to(device).eval()

# load labels mapping if available (expects {"id2label": {"0": "apple_pie", ...}})
LABELS = None
labels_path = os.path.join(os.path.dirname(__file__), 'food101_labels.json')
if os.path.exists(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            if isinstance(data, dict) and 'id2label' in data:
                LABELS = data['id2label']
            elif isinstance(data, dict):
                # fallback: assume dict maps str/int ids to labels directly
                LABELS = {str(k): v for k, v in data.items()}
            print('Loaded labels mapping with', len(LABELS) if LABELS else 0, 'entries')
    except Exception as e:
        print('Failed to load labels file:', e)
else:
    print('No labels file found at', labels_path)

def predict(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        topk = torch.topk(probs, k=5)
        results = {}
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            label = LABELS.get(str(idx), f'class_{idx}') if LABELS is not None else f'class_{idx}'
            results[label] = float(score)
        return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    title='Food101 ResNet50',
    description='Upload food image for classification'
)


if __name__ == '__main__':
    # listen on all interfaces so the container can expose the service
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
