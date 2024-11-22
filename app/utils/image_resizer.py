from flask import Flask
from flask_pymongo import PyMongo
from PIL import Image
import os

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb://localhost:27017/agritest'
mongo = PyMongo(app)

# Read the image file
def read_image(file_path):
    with open(file_path, "rb") as file:
        return file.read()

# Resize the image
def resize_image(image_path, new_width, new_height):
    img = Image.open(image_path)
    resized_img = img.resize((new_width, new_height))
    resized_img.save(image_path)

# Upload images, names, and URLs into the database
def upload_image(image_data, image_name, image_url):
    with app.app_context():
        image_record = {
            "image_data": image_data,
            "image_name": image_name,
            "image_url": image_url
        }
        mongo.db.fertilizer_images.insert_one(image_record)

# Example usage
folder_path = "../static/images/fertilizer_images"  # Path to the folder containing images
new_width = 250
new_height = 300

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        
        # Resize the image
        resize_image(image_path, new_width, new_height)
        
        # Read the resized image
        image_name = os.path.splitext(filename)[0]
        image_url = "https://www.coromandel.biz/product-service/gromor-" + image_name 
        image_data = read_image(image_path)
        
        # Upload the image to MongoDB
        upload_image(image_data, image_name, image_url)
        print(f"Resized and uploaded {filename} to MongoDB")

if __name__ == '__main__':
    app.run(debug=True)

