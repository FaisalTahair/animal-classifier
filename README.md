# animal-classifier
A deep learning-based model for classifying animals into different categories using MobileNetV2. This project includes the trained model, a label encoder, and the steps to use the system for predictions.
## Repository Contents
- `animal_classifier.keras`: The trained model file.
- `label_encoder.pkl`: The LabelEncoder object used for class mapping.
- `animal-classifier.ipynb`: The Jupyter Notebook containing the code for model training, evaluation, and predictions.
- `README.md`: Documentation for the project.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/animal-classifier.git
   cd animal-classifier


Ensure you have the following dependencies installed:

TensorFlow

NumPy

scikit-learn

Matplotlib

Jupyter Notebook

# *Using the Trained Model*
```markdown
3. Open the `animal-classifier.ipynb` notebook in Jupyter or Colab.
4. Load the saved model and LabelEncoder:
   ```python
   from tensorflow.keras.models import load_model
   import pickle

   # Load the model
   model = load_model('animal_classifier.keras')

   # Load the LabelEncoder
   with open('label_encoder.pkl', 'rb') as file:
       label_encoder = pickle.load(file)



import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess the image
img_path = 'path-to-image.jpg'  # Replace with the actual image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)
predicted_class = label_encoder.classes_[np.argmax(prediction)]
print(f'Predicted Class: {predicted_class}')

