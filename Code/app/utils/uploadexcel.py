from flask import Flask, request
import pandas as pd
from flask_pymongo import PyMongo

app = Flask(__name__)

# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/agritest'
mongo = PyMongo(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    # Read the Excel file using pandas
    df = pd.read_excel(file)

    # Iterate over the rows of the DataFrame and insert data into the MongoDB collection
    for index, row in df.iterrows():
        document = {
            'disease_name': row['disease_name'],
            'disease_cause': row['disease_cause'],
            'chemical_methods': row['chemical_methods'],
            'natural_methods': row['natural_methods'],
            'diseases': row['diseases']
        }
        mongo.db.disease_preventionlist.insert_one(document)

    return 'File uploaded successfully', 200

if __name__ == '__main__':
    app.run(debug=True)
