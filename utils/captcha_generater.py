from captcha.image import ImageCaptcha
import random
import os
from tqdm import trange
from config_util import configGetter
from PIL import Image, ImageDraw, ImageFont
import string

cfg = configGetter('DATASET')

image = ImageCaptcha(fonts=[cfg['CAPTCHA']['FONT_DIR']])

def randomSeqGenerator(captcha_len):
    ret = ""
    for i in range(captcha_len):
        num = chr(random.randint(48,57))  # ASCII for numbers
        letter = chr(random.randint(97, 122))  # lowercase letters
        Letter = chr(random.randint(65, 90))  # uppercase letters
        s = str(random.choice([num,letter,Letter]))
        ret += s
    return ret
    
def generate_centered_captcha(text, width=200, height=50):
    # Create a larger white background to contain the rotated text
    padding = 30  # Add padding to ensure rotated text isn't cut off
    temp_image = Image.new('RGBA', (width + padding*2, height + padding*2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_image)

    # Load font Times New Roman
    try:
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Linux path
        font = ImageFont.truetype(font_path, 30)
    except:
        font = ImageFont.load_default()

    # Reduce space between characters
    char_spacing = 16

    # Calculate total width based on number of characters
    total_text_width = len(text) * char_spacing
    start_x = (width + padding*2 - total_text_width) // 2

    # Get actual text dimensions
    left, top, right, bottom = font.getbbox(text)
    text_height = bottom - top
    
    # Place text in the center
    start_y = (height + padding*2 - text_height) // 2 - 10

    # Draw each character
    x = start_x
    for char in text:
        draw.text((x, start_y), char, font=font, fill='black')
        x += char_spacing

    # Rotate the entire image 7 degrees
    rotation_angle = 7
    rotated_text = temp_image.rotate(rotation_angle, resample=Image.BICUBIC, expand=0)
    
    # Create final image
    final_image = Image.new('RGB', (width, height), 'white')
    
    # Insert rotated image to final image, centered
    paste_x = (width - (width + padding*2)) // 2
    paste_y = (height - (height + padding*2)) // 2
    final_image.paste(rotated_text, (paste_x, paste_y), rotated_text)
    
    draw = ImageDraw.Draw(final_image)
    
    # Add plus signs (+ made of 5 dots)
    for _ in range(60):
        x = random.randint(1, width-2)
        y = random.randint(1, height-2)
        # Draw plus sign with 5 points
        draw.point((x, y), fill=(0, 0, 0))      # Center point
        draw.point((x+1, y), fill=(0, 0, 0))    # Right point
        draw.point((x-1, y), fill=(0, 0, 0))    # Left point
        draw.point((x, y+1), fill=(0, 0, 0))    # Bottom point
        draw.point((x, y-1), fill=(0, 0, 0))    # Top point
    
    # Add blurry black dots
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        # Use light gray to create blur effect
        gray_level = random.randint(100, 180)
        draw.point((x, y), fill=(gray_level, gray_level, gray_level))

    # Add noise lines
    for _ in range(12):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill='black', width=1)ter

    return final_image

def captchaGenerator(dataset_path, dataset_len, captcha_len=None):
    # Random captcha length between 3 and 5
    if captcha_len is None:
        # captcha_len = random.randint(3, 5)
        captcha_len = 5
    
    os.makedirs(dataset_path, exist_ok=True)
    
    for i in trange(dataset_len):
        # Generate random character sequence
        char_seq = randomSeqGenerator(captcha_len)
        
        # Generate captcha image
        captcha_image = generate_centered_captcha(char_seq)
        
        # Save the image
        save_path = os.path.join(dataset_path, f'{char_seq}.{i}.png')
        captcha_image.save(save_path)

def generateCaptcha():
    TRAINING_DIR = cfg['TRAINING_DIR']
    TESTING_DIR = cfg['TESTING_DIR']
    TRAINING_DATASET_LEN = cfg['TRAINING_DATASET_LEN']
    TESTING_DATASET_LEN = cfg['TESTING_DATASET_LEN']
    CHAR_LEN = cfg['CAPTCHA']['CHAR_LEN']

    # captchaGenerator(TRAINING_DIR, TRAINING_DATASET_LEN, CHAR_LEN)
    # captchaGenerator(TESTING_DIR, TESTING_DATASET_LEN, CHAR_LEN)
    #captchaGenerator('./dataset/test', 20000, 4)
    
    
    captchaGenerator(TRAINING_DIR, TRAINING_DATASET_LEN)
    captchaGenerator(TESTING_DIR, TESTING_DATASET_LEN)

if __name__ == "__main__":
    generateCaptcha()
