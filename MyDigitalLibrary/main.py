from flask import Flask, request, render_template, jsonify
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

from mylibrary import run_book_stacking

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/assets'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Max 2MB upload


@app.route("/")
def index():
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template("index.html", images=image_files)

@app.route("/predict", methods=["GET", "POST"]) 
def predict():


    data = request.get_json()
    img_path = data.get("img_path", "")
    try:
        book_info ,  marked_img , orig_img = run_book_stacking(img_path , 'static/image.png')
        return jsonify({
            "book_info" : book_info ,  
            "marked_img" : marked_img, 
            "orig_img" : orig_img 
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
