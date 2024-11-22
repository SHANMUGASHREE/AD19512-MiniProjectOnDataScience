from flask import Flask
from flask_pymongo import PyMongo
import os

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
        image_record = {
            "image_data": image_data,
            "image_name": image_name,
            "image_url": image_url
        }
        mongo.db.crop_images.insert_one(image_record)

# Example usage
folder_path = "../static/images/crop_images"
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(folder_path, filename)
        image_name = os.path.splitext(filename)[0]
        image_url = "https://en.wikipedia.org/wiki/" + image_name
        image_data = read_image(image_path)
        upload_image(image_data, image_name, image_url)

if __name__ == '__main__':
    app.run(debug=True)
