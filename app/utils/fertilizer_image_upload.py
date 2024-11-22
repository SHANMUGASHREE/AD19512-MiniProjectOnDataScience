from flask import Flask
from flask_pymongo import PyMongo
import os
import numpy as np

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb://localhost:27017/agritest'
mongo = PyMongo(app)

# Read the image file
def read_image(file_path):
    with open(file_path, "rb") as file:
        return file.read()

# Upload images, names, and URLs into the database
def upload_image(image_data, image_name, image_url):
    with app.app_context():
        image_name = int(image_name) if isinstance(image_name, np.int32) else image_name
        image_record = {
            "image_data": image_data,
            "image_name": image_name,
            "image_url": image_url
        }
        mongo.db.fertilizer_images.insert_one(image_record)

# Example usage
folder_path = "../static/images/fertilizer_images"  # Path to the folder containing images
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(folder_path, filename)
        image_name = os.path.splitext(filename)[0]
        image_url = "https://www.coromandel.biz/product-service/gromor-" + image_name 
        image_data = read_image(image_path)
        upload_image(image_data, image_name, image_url)

if __name__ == '__main__':
    app.run(debug=True)


