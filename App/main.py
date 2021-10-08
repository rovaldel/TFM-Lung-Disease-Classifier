from flask import Flask,request, render_template
import numpy as np 
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
from gradCAM import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model('model/Convolutional-Regularmodel.h5')                 


app = Flask(__name__)
# CORS(app)

@app.route('/',methods=['GET' , 'POST'])
def index():

  if request.method == 'POST':
      f = request.files['file']
      f.save('static/'+secure_filename(f.filename))

      format = ['jpg', 'jpeg', 'png']
      if f.filename.split('.')[-1] in format:
        img_path ='static/'+secure_filename(f.filename)

        size = (224,224)
        img_original = load_img(img_path, target_size=size)
        img_gray     = img_original.convert('L')
        img_array    = img_to_array(img_gray)/255.
        img_exp      = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_exp)

        # Class Names
        class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
        i = np.argmax(prediction[0])

        plt.switch_backend('agg')
        # Impresión de la imagen original
        plt.title('Original Image', fontsize=15)
        plt.xticks([])
        plt.yticks([]) 
        plt.imshow(img_original, cmap="bone", label='Input')
        plt.savefig('static/originals/'+secure_filename(f.filename))

        # Aplicación de técnica GradCAM
        last_conv_layer_name = 'conv2d_6'
        cam = GradCAM(model=model, classIdx=i, layerName=last_conv_layer_name) # find the last 4d shape "mixed10" in this case
        heatmap = cam.compute_heatmap(img_exp)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)                 # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
       
        # Imagen original con 3 canales
        img_original_3 = cv2.imread(img_path)
        img_original_3 = cv2.resize(img_original_3, (224, 224))
        img_heat = cv2.addWeighted(heatmap, 0.5, img_original_3, 1, 0)

        # Impresión del mapa de colores + imagen original
        plt.title(f'Chest Finding Prediction: {class_names[i]}', fontsize=15)
        plt.xticks([])
        plt.yticks([]) 
        plt.imshow(img_heat, cmap="bone", label='Output')
        plt.savefig('static/predictions/'+secure_filename(f.filename))

      return render_template('dashboard.html', img = secure_filename(f.filename))

  if request.method == 'GET':
      return render_template('upload.html')
  
if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=4000)