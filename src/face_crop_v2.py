# src/face_crop_v2.py
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

# Initialize face detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False)

def crop_face(image):
    """
    Crops face from PIL image or image path.
    Returns: PIL Image (224x224) or None if no face found
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Detect face
    face_tensor = mtcnn(image)  # returns torch.Tensor [3,H,W] or None
    if face_tensor is None:
        return None

    # Convert Tensor to PIL Image
    face_pil = transforms.ToPILImage()(face_tensor)

    # Resize to 224x224 to ensure consistency
    face_pil = face_pil.resize((224, 224))

    return face_pil
