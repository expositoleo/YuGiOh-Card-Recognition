import os
import requests

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"
OUTPUT_DIR = "data/yugioh_card_images"

IMAGE_TAG = "image_url_cropped"
FORMAT_TAG = "GOAT"
TIMEOUT = 10

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Images will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# Fetch API data
try:
    response = requests.get(API_URL, timeout=TIMEOUT)
    response.raise_for_status()
    card_data = response.json()
    cards = card_data['data']
    print(f"Found {len(cards)} cards in total from API")
except Exception as e:
    print(f"Failed to fetch card data: {e}")
    cards = []

# Filter cards by format
if FORMAT_TAG is not None:
    cards = [
        card for card in cards
        if any(FORMAT_TAG in format for info in card.get("misc_info", []) for format in info.get("formats", []))
    ]
    print(f"Filtered {len(cards)} cards in {FORMAT_TAG} format.")

# Count total images
total_images = sum(len(card['card_images']) for card in cards)
print(f"Total GOAT images to download: {total_images}")

def download_cards(cards):
    downloaded_count = 0
    errors = 0

    for card in cards:
        for image_info in card['card_images']:
            image_id = image_info['id']
            image_url = image_info[IMAGE_TAG]
            file_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")

            try:
                with requests.get(image_url, stream=True, timeout=TIMEOUT) as img_res:
                    img_res.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in img_res.iter_content(chunk_size=8192):
                            f.write(chunk)
                downloaded_count += 1
                print(f"Downloaded ({downloaded_count}/{total_images}): {image_id}.jpg")
            except Exception as e:
                errors += 1
                print(f"Failed to download image {image_id}: {str(e)}")

            # time.sleep(REQUEST_DELAY)

    print("Download completed!")
    print(f"Total images: {total_images}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Errors: {errors}")
    print(f"Images saved in: {os.path.abspath(OUTPUT_DIR)}")

download_cards(cards)
