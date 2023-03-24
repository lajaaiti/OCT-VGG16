import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model('vgg16-retinal-oct.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    result = classes[np.argmax(predictions)]
    precision = round(np.max(predictions)*100, 2)
    return render_template('index.html', result=result, image=os.path.join(app.config['UPLOAD_FOLDER'], filename), precision=precision)


if __name__ == '__main__':
    app.run(debug=True)
