from flask import Flask, request, jsonify
import os
import base64
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load pre-trained FaceNet model and MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image_cropped = mtcnn(image)
    return image_cropped

def get_embedding(model, image_tensor):
    with torch.no_grad():
        embedding = model(image_tensor.unsqueeze(0))
    return embedding

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

def is_match(embedding1, embedding2, threshold=0.6):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity > threshold

@app.route('/upload-images', methods=['POST'])
def upload_images():
    data = request.get_json()
    image1_base64 = data.get('image1')
    image2_base64 = data.get('image2')

    if not image1_base64 or not image2_base64:
        return jsonify({"error": "Please provide both images in Base64 format"}), 400

    try:
        image1_data = base64.b64decode(image1_base64)
        image2_data = base64.b64decode(image2_base64)

        # Preprocess the images
        image1_tensor = preprocess_image(image1_data)
        image2_tensor = preprocess_image(image2_data)

        if image1_tensor is None or image2_tensor is None:
            return jsonify({"error": "Face not detected in one or both images"}), 400

        # Generate embeddings
        embedding1 = get_embedding(model, image1_tensor)
        embedding2 = get_embedding(model, image2_tensor)

        # Check if the images match
        match = is_match(embedding1, embedding2)

        return jsonify({"match": match}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
