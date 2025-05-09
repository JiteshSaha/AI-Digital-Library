import streamlit as st
from PIL import Image
import tempfile
from mylibrary import run_book_stacking  # Adjust the import as needed

st.set_page_config(page_title="📚 AI-Digital-Library", layout="wide")
st.title("🤖📚 AI-Digital-Library")
st.write("Upload a photo of your bookshelf and let AI do the work!")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_image_path = tmp_file.name

    # Split into two columns
    col1, col2 = st.columns([1, 2])  # Adjust width ratio as needed

    with col1:
        st.image(Image.open(tmp_image_path), caption="📸 Uploaded Bookshelf", use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing bookshelf..."):
            book_info, detected_img = run_book_stacking(tmp_image_path)

        st.success("✅ Detection complete!")

        if book_info:
            st.subheader(f"📖 Identified Books: {len(book_info)}")
            st.image(Image.open('image.png'), caption="📸 Detected Books", use_container_width=False)
            for book in book_info:

                if book['title']:
                    st.markdown(f"""
                        ---
                        📘 **Book ID:** {int(book['id']) + 1}  
                        📖 **Title** ({book.get('title_similarity', 0):.2f}): `{book.get('title', '')}`  
                        ✍️ **Author:** `{book.get('author', 'Unknown')}`  
                        🔍 **Confidence:** {book['confidence']}% \n
                        🔍 **RAW:** {", ".join(book['Raw OCR'])}
                        """)

    
        else:
            st.warning("⚠️ No books identified.")
