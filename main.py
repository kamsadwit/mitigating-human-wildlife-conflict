from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import torch
from PIL import Image
import cv2  # Import OpenCV

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/inference_output/'
# app.config['PROCESSED_FOLDER'] = 'static/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('process_file', filename=filename))
        
        flash('Invalid file type')
        return redirect(request.url)

    return render_template('index.html')

@app.route('/process/<filename>')
def process_file(filename):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.25  # Set confidence threshold

    # Load the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(img_path)

    # Perform inference
    results = model(img)
     # Get the annotated image
    annotated_image = results.render()[0]  # Get the image with bounding boxes

    # Convert the image from RGB (PIL) to BGR (OpenCV)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Display the image in a new window using OpenCV
    # cv2.imshow('Processed Image', annotated_image)
    cv2.imshow('Processed Image', annotated_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    # Save the annotated image
    results.save(save_dir=app.config['PROCESSED_FOLDER'])
    print(app.config['PROCESSED_FOLDER'] + filename)
    
    # # Get the latest saved image from the processed folder
    # image_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if os.path.isfile(os.path.join(app.config['PROCESSED_FOLDER'], f))]
    # # latest_image = sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(app.config['PROCESSED_FOLDER'], x)))[0]
    # print(sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(app.config['PROCESSED_FOLDER'], x))))
    # latest_image_path = os.path.join(app.config['PROCESSED_FOLDER'], latest_image)
    




    # # Print relative and absolute paths
    # print(f"Relative path: {os.path.relpath(latest_image_path)}")
    # print(f"Absolute path: {os.path.abspath(latest_image_path)}")

    # Get URLs for displaying in the template
    processed_image_url = url_for('static', filename=f'inference_output/{filename}')

    return render_template('result.html', 
                           selected_image_url=url_for('uploaded_file', filename=filename),
                           processed_image_url=processed_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
