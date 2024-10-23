import os
import google.generativeai as genai
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/efficientnet_palm_leaf_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'  # For flash messages

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names corresponding to the palm leaf conditions
class_names = ['Black Scorch', 'Dubas', 'Bug', 'Fusarium Wilt', 'Healthy sample', 'Honey', 'Leaf Spots',
               'Magnesium Deficiency', 'Manganese Deficiency', 'Parlatoria Blanchardi', 
               'Potassium Deficiency', 'Rachis Blight']

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function for uploaded images
def predict_image(img_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values

    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

# Function to query the Gemini API using the `google-generativeai` library
def query_gemini_api(deficiency):
    # Configure Gemini API key and generation settings
    genai.configure(api_key="xx")
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Create the model instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session and send the deficiency as input
    chat_session = model.start_chat(
        history=[]
    )
    
    # Send deficiency query to the model
    response = chat_session.send_message(f" {deficiency} disease in palm tree.")

    # Extract the text content from the response
    response_text = response.text  # Assuming 'text' holds the output

    # Split the response text based on '**' and return as a list
    r = response_text.split("**")
    s=" ".join(i for i in r)

    # Return the text response
    return s


# Route to upload and predict image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, browser may submit an empty part without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # If the file is allowed, save it and make a prediction
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict the class of the image (deficiency)
            predicted_class = predict_image(file_path)
            
            # Call Gemini API with the predicted deficiency
            gemini_data = query_gemini_api(predicted_class)
            
            # Return the result to the user
            return render_template('result.html', predicted_class=predicted_class, gemini_data=gemini_data, filename=filename)
    
    # If GET request, show the upload form
    return render_template('upload.html')

# Route to display uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
