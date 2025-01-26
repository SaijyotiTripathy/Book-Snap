import os
import cv2
import numpy as np
import requests
import json
import base64
from PIL import Image
from dotenv import load_dotenv
from inference import get_model
import pandas as pd
from SeleniumHandler import SeleniumHandler
from urllib.request import urlparse, quote
from selenium.webdriver.common.by import By
from tqdm import tqdm
import json
from selenium.common.exceptions import NoSuchElementException
import time

tqdm.pandas()

# Load environment variables
load_dotenv()


# Segment books from images using finetuned yolo model
class BookSegmentationModel:
    def __init__(self, model_id, api_key):
        self.model = get_model(model_id=model_id, api_key=api_key)

    def segment_books(self, image_path):
        segmented_books = []
        results = self.model.infer(image_path)

        predictions = results[0].predictions
        image = Image.open(image_path)
        image_np = np.array(image)

        segmentations_dir = os.path.join("temp", "segmentations")
        os.makedirs(segmentations_dir, exist_ok=True)

        for prediction in predictions:
            points = [(point.x, point.y) for point in prediction.points]
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            polygon = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 255)

            masked_region = cv2.bitwise_and(image_np, image_np, mask=mask)
            non_zero_coords = np.argwhere(mask > 0)
            top_left = non_zero_coords.min(axis=0)
            bottom_right = non_zero_coords.max(axis=0)
            segmented_cropped = masked_region[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

            segmented_image_path = os.path.join(segmentations_dir, f"segmented_{prediction.detection_id}.png")
            segmented_cropped_img = Image.fromarray(segmented_cropped)
            segmented_cropped_img.save(segmented_image_path)

            segmented_books.append({
                "image_name": image_path.split("/")[-1],
                "detection_id": prediction.detection_id,
                "confidence": prediction.confidence,
                "points": points,
                "segmented_image": segmented_cropped,
                "segmented_image_path": segmented_image_path
            })

        return segmented_books

# Process the segmented book images using OpenAI API
class OpenAIProcessor:
    def __init__(self, api_key):
        self.api_key = api_key

    def encode_image(self, image):
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    def process_mask(self, image, text):
        base64_image = self.encode_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if "choices" in response_data:
            result = response_data["choices"][0]["message"]["content"].rstrip("Title:").replace("\n", " ")
            return result
        return None


# Google Scraper to get book links
class GoogleScraper:
    @staticmethod
    def search_google(keyword):
        sel = SeleniumHandler()
        sel.driver.get(f"https://www.google.com/search?as_q={"+".join(word for word in  keyword.split(" "))}")
        elements = [element for element in sel.driver.find_elements(By.CLASS_NAME, "MjjYud")]
        links = []
        for element in elements:
            a_tags = element.find_elements(By.TAG_NAME, "a")
            if len(a_tags) > 0:
                links.append(a_tags[0].get_attribute("href"))
        sel.driver.quit()
        return links
    

# Search Goodreads link for book details
class Goodreads(SeleniumHandler):
    def __init__(self):
        super().__init__()

    def scrap(self, link):
        self.driver.get(link)
        info={}

        # Get Title, Authors, Description, Genres, Reviews
        element = self.driver.find_element(By.CLASS_NAME, "BookPageTitleSection")
        info["Title"]= element.text

        elements = self.driver.find_elements(By.XPATH, """/html/body/div[1]/div[2]/main/div[1]/div[2]/div[2]/div[2]/div[1]/h3/div/span[1]""")
        authors=''
        for element in elements:
            authors+= element.text.replace("\n"," ")
        info["Authors"]= authors

        element= self.driver.find_element(By.CSS_SELECTOR, 'div[data-testid="description"]')
        desc= element.text
        desc= desc.replace("\n", " ")
        desc= desc.replace("Show more", "")
        info["Description"]= desc

        element= self.driver.find_element(By.CSS_SELECTOR, 'div[data-testid="genresList"]')
        genres= element.text
        genres= genres.replace("Genres\n","")
        genres= genres.replace("...more", "")
        genres= genres.split('\n')
        info["Genres"]= genres

        time.sleep(5)

        review_cards = self.driver.find_elements(By.CLASS_NAME, "ReviewCard")
        reviews = []
        for count, card in enumerate(review_cards):
            if count == 3:
                break
            try:
                content = card.find_element(By.CLASS_NAME, "ReviewCard__content")
                review_text = content.find_element(By.CLASS_NAME, "ReviewText").text
                review_text= review_text.replace("\n", " ")
                review_text= review_text.replace("Show more", "")
                reviews.append(review_text)
            except NoSuchElementException:
                continue
        for count,review in enumerate(reviews):
            info[f"Review {count+1}"]= review
      
        print(info)
        return info


# Class to handle the entire book detection and data retrieval process
class BookProcessor:
    def __init__(self, segmentation_model, openai_processor, google_scraper, goodreads):
        self.segmentation_model = segmentation_model
        self.openai_processor = openai_processor
        self.google_scraper= google_scraper
        self.goodreads = goodreads
        self.results = {}
        self.book_details= {}

    def process_books(self, image_path):
        text_prompt = "Read the text on the book spine. Only say the book cover title and author if you can find them. Return the format [title]-[author], with hypen in between."
        segmented_books = self.segmentation_model.segment_books(image_path)
        # Save openai results
        for book in segmented_books:
            result = self.openai_processor.process_mask(book["segmented_image"], text_prompt)
            if result:
                self.results[result] = {
                    "openai_result": result,
                    "book_image_path": book["segmented_image_path"],
                    "segmented_points": book["points"]
                }
        with open("temp/all_details.json", "w") as f:
            json.dump(self.results, f, indent=4)

        
        for book, book_data in self.results.items():
            openai_result= book_data["openai_result"]
            book_name= ' '.join(openai_result.split(' '))
            print(book_name)
            link_list= self.google_scraper.search_google(f"{book_name} goodreads")
            for link in link_list:
                if link.startswith("https://www.goodreads.com/"):
                    print(link)
                    info= self.goodreads.scrap(link)
                    info["Goodreads link"]= link
                    self.book_details[book_name]= info
                    break
            time.sleep(5)
        
        with open('book_details.json','w') as f:
            json.dump(self.book_details, f, indent=4)





