from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Load pre-trained model
model = load_model('final_model.h5')

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
    img = Image.open(io.BytesIO(image))
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your React.js app runs on this port
    "http://localhost:3001",  # Assuming your FastAPI app runs on this port
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()

        # Preprocess the image
        preprocessed_image = preprocess_image(contents)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100

        # Prepare response
        class_probabilities = {class_labels[i]: str(prediction[0][i] * 100) for i in range(len(class_labels))}

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "confidence": str(confidence),
                "class_probabilities": class_probabilities
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
