import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(model, image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prob_fake = torch.softmax(output, dim=1)[0, 0].item()

    label = "FAKE" if prob_fake > 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else (1 - prob_fake)

    return label, confidence
