import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
from transformers import T5Tokenizer, T5ForConditionalGeneration


from logger import setup_logger

logger = setup_logger()


# ---------------------------- Detection Module ----------------------------

API_KEY = r""

def run_book_spine_detection(img_path):
    model = YOLO('yolov8n.pt')  # Replace with custom weights if needed
    results = model(img_path)   # Inference
    return results

import matplotlib.pyplot as plt
import cv2

def visualize_detections(result, book_info, save_path="image.png"):

    logger.debug(result)
    orig_img = result.orig_img

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

    book_count = 0

    for idx, box in enumerate(book_info):
                
            x1, y1, x2, y2 = box['box']
            conf = box['confidence']

            label = f"Book {int(box['id'])+1}: {conf:.2f}"
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', lw=2)
            ax.text(x1, y1 - 10, label, color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))

            book_count += 1
        # if names[cls] == 'book':

    logger.info(f"📚 Total detected books: {book_count}")
    plt.axis('off')
    
    # Save figure to file
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory

    # return save_path



# def visualize_detections(result, save_path="image.png"):

#     logger.debug(result)
#     boxes = result.boxes
#     names = result.names
#     orig_img = result.orig_img

#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

#     book_count = 0

#     for idx, box in enumerate(boxes):
        
        
#         cls_id = int(box.cls[0])
#         cls_name = names[cls_id]

#         logger.info(cls_name)
#         if cls_name == 'book':
        
#             x1, y1, x2, y2 = box.xyxy[0]
#             conf = box.conf[0]
#             cls = int(box.cls[0])

#             label = f"Book {idx+1}: {conf:.2f}"
#             ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', lw=2)
#             ax.text(x1, y1 - 10, label, color='white', fontsize=12,
#                 bbox=dict(facecolor='red', alpha=0.5))

#             book_count += 1
#         # if names[cls] == 'book':

#     logger.info(f"📚 Total detected books: {book_count}")
#     plt.axis('off')
    
#     # Save figure to file
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)  # Close the figure to free memory

#     # return save_path




def extract_book_regions(results, target_class='book'):
    regions = []
    result = results[0]  # Assume one image
    boxes = result.boxes
    names = result.names
    image = result.orig_img

    h, w, _ = image.shape

    count = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        logger.info(cls_name)
        if cls_name == target_class and float(box.conf[0]) > 0.30 :
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounds check
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            regions.append({
                'id' : count,
                'coordinates': (x1, y1, x2, y2),
                'class': cls_name,
                'confidence': float(box.conf[0]),
                'image': image
            })

            count += 1

    return regions


# ---------------------------- OCR Module ----------------------------

reader = easyocr.Reader(['en'])  # Load once


def run_easyocr_on_regions(regions):
    ocr_results = []

    for idx, region in enumerate(regions):
        x1, y1, x2, y2 = region['coordinates']
        image = region['image']
        crop = image[y1:y2, x1:x2]

        result_text = reader.readtext(crop, detail=0)

        ocr_results.append({
            'id': region['id'],
            'text': result_text,
            'confidence': region['confidence'],
            'box': region['coordinates']
        })

    return ocr_results


# ---------------------------- T5 Text Parsing Module ----------------------------

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)


def read_titles(ocr_list):
    ocr_text = ", ".join(ocr_list)

    prompt = f"""
Given the following text detected from a book spine, extract the title and the author.

Text: {ocr_text}

Return the output as:
Title: <book title>
Author: <author name>
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = t5_model.generate(**inputs, max_length=64)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def parse_t5_output(text):
    result = {}
    for line in text.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            result[key.strip().lower()] = val.strip()
    return result

from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_book_title(ocr_list):

    # Load the model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Ensure OCR list contains only strings
    cleaned_ocr_list = [word for word in ocr_list if isinstance(word, str)]

    # Join the words into one string
    raw_text = " ".join(cleaned_ocr_list).strip()

    # If the OCR text is empty, return a default message
    if not raw_text:
        return "Title not detected"

    # Format the input prompt
    prompt = f"""
    You are a book expert. Given the following noisy or partial text from the spine of a book, guess the most probable book title.

    Text: {raw_text}

    Book Title:
    """    # Format the input prompt
    test_prompt = f"""
        just repeat the same exact thing: {raw_text}
    """

    # Tokenize input
    inputs = tokenizer(test_prompt.strip(), return_tensors="pt", truncation=True)

    # Generate the output with num_beams for better quality
    output = model.generate(**inputs, max_length=32, num_beams=5, num_return_sequences=1, early_stopping=True)

    # Decode the output and return the result
    generated_title = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return generated_title


import openai

# Function to generate book title using GPT-3 or GPT-4
def generate_book_title_with_gpt(ocr_list):
    # Combine OCR list into a single string
    raw_text = " ".join(map(str, ocr_list))

    # Set up the OpenAI API key (replace 'your-api-key' with your actual key)
    openai.api_key = "your-api-key"  # Add your OpenAI API key here

    # Prompt to ask GPT to generate the book title
    prompt = f"""
    You are a book expert. The following text is from the spine of a book, but it's noisy and partial. Please guess the most probable book title based on this text:

    Text: {raw_text}

    Book Title:
    """

    try:
        # Send the prompt to the GPT model (e.g., GPT-3 or GPT-4)
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the appropriate engine (davinci is for GPT-3, GPT-4 is another option)
            prompt=prompt.strip(),
            max_tokens=64,
            n=1,
            temperature=0.7,  # Controls randomness; adjust as necessary
            stop=["\n"]  # Ensure the output stops when the book title is generated
        )

        # Get the book title from the response
        book_title = response.choices[0].text.strip()
        return book_title

    except Exception as e:
        logger.info(f"Error: {e}")
        return None


import requests
import requests
from fuzzywuzzy import fuzz
import requests
from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(query, title):
    vectorizer = TfidfVectorizer().fit([query, title])
    tfidf_matrix = vectorizer.transform([query, title])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]



def search_google_books(query_list, api_key= API_KEY):
    query = " ".join(query_list)

    # url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={api_key}"
    
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        if not items:
            logger.info("No books found.")
            return None

        ranked_books = []

        for item in items:
            title = item["volumeInfo"].get("title", "No title")
            authors = item["volumeInfo"].get("authors", ["No author"])
            author = ", ".join(authors)

            # title_similarity = fuzz.partial_ratio(query.lower(), title.lower())
            title_similarity = compute_cosine_similarity(query, title)

            ranked_books.append({
                "title": title,
                "author": author,
                "title_similarity": title_similarity
            })


        # Sort by similarity
        ranked_books.sort(key=lambda x: x["title_similarity"], reverse=True)

        # Return top book
        top_book = ranked_books[0]
        # logger.info("🎯 Best Match:")
        # logger.info(f"📖 Title: {top_book['title']}")
        # logger.info(f"✍️ Author: {top_book['author']}")
        # logger.info(f"📊 Title Similarity: {top_book['title_similarity']}")

        
        logger.info(f'\n{query=}')
        logger.info(f'{ranked_books=}')
        return top_book

    else:
        logger.info(f"❌ Failed to fetch data. Status code: {response.status_code}")
        return {}

# # Example OCR text (You can use your OCR output here)
# ocr_text = "justice league society"
# api_key = "YOUR_GOOGLE_BOOKS_API_KEY"  # Replace this with your actual API key

# search_google_books(ocr_text, api_key)


def get_books(ocr_output):
        # Step 4: NLP extraction (Title + Author)
    books = []
    for item in ocr_output:
        # logger.info(f"🔲 Box: {item['box']}")
        book_info = search_google_books(item['text'])
   

        # logger.info(f'{book_info=}')
        books.append({
            'id': item['id'],
            'confidence': round(item['confidence'] * 100, 2),
            'Raw OCR' : (item['text']),
            'title': book_info.get('title', item['text']) if book_info else None,
            'author': book_info.get('author', '') if book_info else None,
            'title_similarity': book_info.get('title_similarity', '') if book_info else None,
            'box' : item['box']
        })

    return books
# ---------------------------- Main Pipeline ----------------------------


def run_book_stacking(image_path):

   
    # Step 1: Detect books
    results = run_book_spine_detection(image_path)
    

    # Step 2: Extract bounding boxes for books
    book_regions = extract_book_regions(results)

    logger.info(f'{book_regions=}')

    # Step 3: OCR on cropped regions
    ocr_output = run_easyocr_on_regions(book_regions)

    book_info = get_books(ocr_output)

    return book_info , visualize_detections(results[0], book_info)



if __name__ == "__main__":

    image_path = "assets/my-books.png"


    book_info, _ = run_book_stacking(image_path)
    logger.debug(f'{book_info=}')
    for book in book_info:
        logger.info(f"\n📘 Region {book['id']} ({book['confidence']}%)")
        logger.info(f"📖 Title ({book.get('title_similarity','')}) = {book.get('title','')}")
        logger.info(f"📖 Author = {book.get('author','')}")