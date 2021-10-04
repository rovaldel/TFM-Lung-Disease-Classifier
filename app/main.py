from flask import Flask,request, render_template
import numpy as np 
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models  import load_model
import cv2
from gradCAM import GradCAM
import tensorflow as tf

model = load_model('model/Convolutional-Regularmodel.h5')                 


app = Flask(__name__)
# CORS(app)

@app.route('/',methods=['GET' , 'POST'])
def index():

  if request.method == 'POST':
      f = request.files['file']
      f.save('static/'+secure_filename(f.filename))

      if f.filename.split('.')[-1] == 'jpeg' or f.filename.split('.')[-1] == 'png':
        img_path ='static/'+secure_filename(f.filename)

        size = (224,224)
        img_original = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        img_gray     = img_original.convert('L')
        img_array    = tf.keras.preprocessing.image.img_to_array(img_gray)/255.
        img_exp      = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_exp)

        # Class Names
        class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
        i = np.argmax(prediction[0])


        # Impresión de la imagen original
        plt.title('Original Image', fontsize=15)
        plt.xticks([])
        plt.yticks([]) 
        plt.imshow(img_original, cmap="bone", label='Input')
        plt.savefig('static/originals/'+secure_filename(f.filename))


        # Aplicación de técnica GradCAM
        last_conv_layer_name = 'conv2d_6'
        cam = GradCAM(model=model, classIdx=i, layerName=last_conv_layer_name) # find the last 4d shape "mixed10" in this case
        heatmap  = cam.compute_heatmap(img_exp)
        heatmap  = cv2.resize(heatmap, (224, 224))
        heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)               # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
       
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
  app.run(threaded=False)