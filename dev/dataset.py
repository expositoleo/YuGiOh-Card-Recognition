import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

IMAGES_DIR = "data/yugioh_card_images"

class CustomImageDataset(Dataset):
    def __init__(self, image_dir=IMAGES_DIR, augment_prob=0.5, transform=None):
        self.image_dir = image_dir
        self.augment_prob = augment_prob
        self.image_files = [
            f for f in os.listdir(image_dir)
        ]
        self.transform = transform

    def _apply_augmentations(self, img):
        """Apply distortions to images"""
        
        # Apply blur (camera out-of-focus or motion)
        if random.random() < self.augment_prob:
            blur_types = [ImageFilter.GaussianBlur, ImageFilter.BoxBlur]
            blur = random.choice(blur_types)(radius=random.uniform(1.0, 5.0))  # increased blur radius
            img = img.filter(blur)
        
        # Apply lighting adjustments
        if random.random() < self.augment_prob:
            # Brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))  # slightly wider range
            
            # Contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        
        # Apply color shifts
        if random.random() < self.augment_prob:
            # Color balance
            r, g, b = img.split()
            factors = [random.uniform(0.7, 1.3) for _ in range(3)]  # wider shifts
            r = r.point(lambda i: i * factors[0])
            g = g.point(lambda i: i * factors[1])
            b = b.point(lambda i: i * factors[2])
            img = Image.merge("RGB", (r, g, b))

        # Add sensor noise
        if random.random() < self.augment_prob:
            img = self._add_realistic_noise(img)
        
        return img

    def _add_realistic_noise(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Poisson noise (photon shot noise)
        # Increase photon count to amplify noise
        poisson_noise = np.random.poisson(img_array * 20) / 20.0
        img_array = 0.5 * img_array + 0.5 * poisson_noise  # balance more towards noise
        
        # Gaussian noise (sensor read noise)
        gaussian_noise = np.random.normal(0, 0.05, img_array.shape)  # higher std dev
        img_array += gaussian_noise
        
        # Clip and convert back
        img_array = np.clip(img_array, 0, 1) * 255
        return Image.fromarray(img_array.astype(np.uint8))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image_positive = self._apply_augmentations(image)
        while True:
            negative_filename = self.image_files[random.randint(0, len(self.image_files) - 1)]
            if negative_filename != filename:
                negative_card_id = negative_filename.split(".")[0]
                image_negative = Image.open(os.path.join(self.image_dir, negative_filename)).convert("RGB")
                break

        
        if self.transform:
            image = self.transform(image)
            image_positive = self.transform(image_positive)
            image_negative = self.transform(image_negative)
            
        card_id = filename.split(".")[0]

        return image, image_positive, image_negative, card_id, negative_card_id

# transform = transforms.Compose([
#     transforms.Resize((255,255)),
#     transforms.ToTensor()
# ])

# dataset = CustomImageDataset(image_dir=os.path.abspath(IMAGES_DIR), transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# for images, images_pos, images_neg, ids, neg_ids in dataloader:
#     print(images.shape)  # [batch, 3, 255, 255]
#     print(ids)           # IDs string list
#     print(neg_ids)       # Negative IDs string list
#     break