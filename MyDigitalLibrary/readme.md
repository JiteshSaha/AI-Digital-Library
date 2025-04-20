Sure! Here's a clean and professional `README.md` you can use for your **AI-Digital-Library** project:

---

```markdown
# 📚 AI-Digital-Library

AI-Digital-Library is a smart tool that scans an image of a bookshelf, detects books from their spines, reads the visible text using OCR, and attempts to identify book titles and authors using the Google Books API. It's a perfect starter project for digitizing personal libraries or exploring computer vision + NLP pipelines.

---

## 🚀 Features

-  Upload a bookshelf photo
-  YOLOv8-based book spine detection
-  EasyOCR for extracting spine text
-  NLP/Transformer model (T5) or Google Books API for identifying titles
-  Ranks and displays most likely book titles with confidence
-  Visualizes book detection boxes on the image
-  Simple UI built using Streamlit

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/ai-digital-library.git
   cd ai-digital-library
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **[Optional] Set up Google Books API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create an API key for the **Books API**
   - Save your key in a `.env` file:
     ```
     GOOGLE_BOOKS_API_KEY=your_api_key_here
     ```

---

## 🧪 Run the App

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit UI
├── library.py              # Main processing logic (detection, OCR, API)
├── utils.py                # Helper functions (image saving, visualization)
├── requirements.txt
├── README.md
└── assets/
    └── sample_bookshelf.jpg
```

---

## 🛠 Tech Stack

- **YOLOv8** – Object detection
- **EasyOCR** – OCR for text extraction
- **T5 Transformer / Google Books API** – Title guessing
- **FuzzyWuzzy** – Text similarity scoring
- **Streamlit** – Frontend UI
- **OpenCV / Matplotlib** – Image processing

---

## 📸 Example

Upload an image like this:

![example](assets/sample_bookshelf.png)

And get results like:

```
📘 Region 2 (92.15%)
📖 Title (93) = The Catcher in the Rye
✍️ Author = J.D. Salinger
```

---

## 📄 License

MIT License. Free to use and modify!

---

## 🙌 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [HuggingFace Transformers](https://huggingface.co/)
- [Google Books API](https://developers.google.com/books)

---

## 💡 Future Ideas

- Save book records to a database
- Export to CSV or Notion
- Support rotated spines / stacked books
- Mobile upload interface

---

Enjoy scanning your shelves! 📚✨
```
