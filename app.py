from flask import Flask, render_template, request, jsonify, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
import string
import tensorflow as tf
from tensorflow import image
import nltk
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
nltk.download('wordnet')
nltk.download('omw-1.4')
import joblib
import tensorflow_text as text
import time




UPLOAD_FOLDER = '/static/imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def product_card(imagename, text, pid, page):
    
    
    if page == 'product':
        a = prodcard.replace('product_link_placeholder', str(pid))
    elif page == 'index':
        a = prodcard.replace('product_link_placeholder', 'product/'+str(pid))
    a = a.replace('image_name_placeholder', 'https://shopeeimages.blob.core.windows.net/newcontainer/'+imagename)
    a = a.replace('text_placeholder', text)
    
    return a

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
image_size = [224,224] #We will convert all the images to 224x224x3 size

def decode_image(image_data):
  '''
  This function takes the location of a image data
  converts it and returns tensors
  '''
  img = tf.io.read_file(image_data)
  img = tf.io.decode_jpeg(img, channels = 3)
  img = image.resize(img, image_size)
  img = tf.cast(img, tf.float32) / 255.0
  img = tf.reshape(img, [-1, 224, 224, 3])
  return img

def preprocessing(text):
  '''
  This function takes raw text Converts uppercase to Lower, Removes special charecters, stop words 
  and text that follows special charecters (Ex \xe3\x80\x80\)
  '''

  text=text.strip() #Strips the sentence into words
  text = text.lower() #Converts uppercase letters to lowercase
  text = text.split(' ') #Splitting sentence to words
  dummy = []
  for i in text:
    if i[:2] == '\\x':
      special_texts = [i.split("\\") for i in text if i[:2] == '\\x']
      for j in special_texts[0]:
        if len(j) > 3:
          dummy.append(j[3:])
    else:
      dummy.append(i)
  text = dummy
  stopwords = get_stop_words('english') + get_stop_words('indonesian') #Getting stopwords from english and indonesian languages
  text = [i for i in text if i not in stopwords] ##Removing stopwords
  wordnet_lemmatizer = WordNetLemmatizer() #Loading Lemmetizer
  text = [wordnet_lemmatizer.lemmatize(word) for word in text] #Lemmetizing the text
  text = " ".join([i for i in text if len(i) < 15]) #Remove the words longer than 15 charecters
  text = "".join([i for i in text if i not in string.punctuation]) #Removing special charecters
  return text

@app.route('/')
def index():
    f = open("templates/index.html", "r")
    f = str(f.read())
    
    return f

@app.route('/load_globals')
def load():
    if 'checkglobal' in globals():
        pass
    else:
        global interpreter
        global images_search_model
        global tx
        global text_search_model
        global input_details
        global output_details
        global imagename_list
        global text_list
        global pid_list
        global checkglobal
        global dataset
        global prodcard
        global prodpage
        global srchpage
        
        interpreter = tf.lite.Interpreter(model_path="image_model.tflite")
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tx = tf.keras.models.load_model('text_model')
        images_search_model = joblib.load('image_search.sav')
        text_search_model = joblib.load('text_search.sav')
        
        dataset = pd.read_csv('test_dataset.csv')
        
        imagename_list = list(dataset.image)
        text_list = list(dataset.title)
        pid_list = list(dataset.posting_id)
        
        prodcard = open("templates/product_card.html", "r")
        prodcard = str(prodcard.read())
        
        prodpage = open("templates/product.html", "r")
        prodpage = str(prodpage.read())
        
        srchpage = open("templates/searchresults.html", "r")
        srchpage = str(srchpage.read())
        

    return 'globals loaded'

@app.route('/product/<string:pid>', methods=['GET'])
def product_page(pid):
    
    pid_data = dataset[dataset.posting_id == pid]
    imagename = str(list(pid_data.image)[0])
    text = str(list(pid_data.title)[0])
    
    f = prodpage.replace('image_name_placeholder', 'https://shopeeimages.blob.core.windows.net/newcontainer/'+imagename)
    f = f.replace('text_placeholder', text)
    
    similarProducts = pid_data.similar_products.values[0]
    similarProducts = similarProducts.strip("][").split(', ')
    similarProducts = [i.strip("'") for i in similarProducts]
    similarProducts = dataset[dataset.posting_id.isin(similarProducts)]
    
    div = ''
    for i in range(len(similarProducts)):
        imagename = list(similarProducts.image)[i]
        text = list(similarProducts.title)[i]
        pid = list(similarProducts.posting_id)[i]
        if len(text) > 50:
            text = text[:50]+'...'
        else:
            text = text + '...' + ' '*(50-len(text))
        div+=product_card(imagename, text, pid, 'product')
    
        
    f = f.replace('products_place_holder', div)
    return f

@app.route('/search', methods=['POST'])
def search_page():
    if request.method == 'POST':
        indices = []
        
        text = request.form['text']
        if len(text) > 0:
            text = preprocessing(text)
            
            t = tx.predict([text])
            text_distances, text_indices = text_search_model.kneighbors(t)
            indices = text_indices[0][:10].tolist()
        
        
        if 'file' not in request.files:
            if len(indices) > 0:
                pass
            
        file = request.files['file']
        if file.filename == '':
            if len(indices) > 0:
                pass
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/imgs/'+filename)
            
            img = decode_image('static/imgs/'+filename)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            l = interpreter.get_tensor(output_details[0]['index'])
            image_distances, image_indices = images_search_model.kneighbors(l)
            
            if len(indices)>0:
                indices = indices[:5]
                indices.extend(image_indices[0][:5].tolist())
            else:
                indices.extend(image_indices[0][:10].tolist())
            
        
        
        div = ''
        if len(indices) == 0:
            f = srchpage.replace('products_place_holder', '<center>Enter atleast a query text or an image to search<center>')
        else:
            for i in indices:
                imagename = imagename_list[i]
                text = text_list[i]
                pid = pid_list[i]
                if len(text) > 50:
                    text = text[:50]+'...'
                else:
                    text = text + '...' + ' '*(50-len(text))
                div+=product_card(imagename, text, pid, 'index')
                
            f = srchpage.replace('products_place_holder', div)
        
        return f
    
    

if __name__ == '__main__':
    app.debug = False
    app.run()
    

