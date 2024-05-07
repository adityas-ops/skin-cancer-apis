from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load pre-trained model
model = load_model('aditya.h5')

# Define class labels
class_labels = {
    4: 'nv',
    6: 'mel',
    2: 'bkl',
    1: 'bcc',
    5: 'vasc',
    0: 'akiec',
    3: 'df'
}

# Function to preprocess image
def preprocess_image(image):
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array_normalized = img_array / 255.0
    img_array_scaled = img_array_normalized.reshape(1, 28, 28, 1)
    return img_array_scaled

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        return JSONResponse(content={"prediction": predicted_class}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
