import os
import json
from dotenv import load_dotenv
import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from backend import BookSegmentationModel, OpenAIProcessor, BookProcessor, GoogleScraper, Goodreads
from io import BytesIO
import base64
import pandas as pd
import time


load_dotenv()

# Helper function to convert an image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Constants
TEMP_FOLDER = "temp"
Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)

# Streamlit UI
st.title("BookSnap – Snap & Read")
st.write("This AI-powered app transforms your camera into a **bibliophile’s dream assistant**, automatically identifying books and fetching their Goodreads details—all in a snap! 📸✨ ")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    image_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display the image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Create an "Enter" button for processing
    if st.button("Get Details"):
        # Initialize model components
        segmentation_api_key = os.getenv("SEGMENTATION_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model_id = "bookstore-cxuck/1"
        
        segmentation_model = BookSegmentationModel(model_id, segmentation_api_key)
        openai_processor = OpenAIProcessor(openai_api_key)
        google_scraper= GoogleScraper()
        goodreads = Goodreads()

        processing_message = st.empty()
        processing_message.write("Processing the image...")

        processor = BookProcessor(segmentation_model, openai_processor, google_scraper, goodreads)

        # Process the uploaded image
        processor.process_books(image_path)
        points = processor.results
        book_details= processor.book_details
        book_details= pd.DataFrame(book_details)

        processing_message.empty()
        st.write("Processing complete.")

        # Create a new image with segmentations (bounding boxes)
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        for book_title, book_info in points.items():
            # Draw bounding box on the image
            points = book_info["segmented_points"]
            x_min, y_min = min([p[0] for p in points]), min([p[1] for p in points])
            x_max, y_max = max([p[0] for p in points]), max([p[1] for p in points])
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        st.image(img_with_boxes, caption="Detected books", use_container_width=True)


        st.write("Detected Book Details:")
        st.dataframe(book_details)


        st.write("You can download the result CSV file below:")
        st.download_button(
            label="Download CSV",
            data= book_details.to_csv(index=True).encode('utf-8'),
            file_name="book_details.csv",
            mime="text/csv",
        )
