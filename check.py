from PIL import Image, ImageDraw, ImageFont
import random
import string
import math

def generate_centered_captcha(text, width=200, height=50):
    # Tạo nền trắng lớn hơn để chứa văn bản đã xoay
    padding = 30  # Thêm padding để đảm bảo văn bản xoay không bị cắt
    temp_image = Image.new('RGBA', (width + padding*2, height + padding*2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_image)

    # Load font Times New Roman
    try:
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Linux path
        font = ImageFont.truetype(font_path, 30)
    except:
        font = ImageFont.load_default()

    # Giảm khoảng cách giữa các ký tự
    char_spacing = 15  # Giảm từ 25 xuống 15

    # Tính tổng chiều rộng dựa trên số ký tự
    total_text_width = len(text) * char_spacing
    start_x = (width + padding*2 - total_text_width) // 2

    # Lấy kích thước thực của văn bản
    left, top, right, bottom = font.getbbox(text)
    text_height = bottom - top
    
    # Đặt văn bản ở chính giữa
    start_y = (height + padding*2 - text_height) // 2 - 10

    # Vẽ từng ký tự
    x = start_x
    for char in text:
        draw.text((x, start_y), char, font=font, fill='black')
        x += char_spacing

    # Xoay toàn bộ ảnh 10 độ sang trái
    rotation_angle = 7  # Nghiêng sang
    rotated_text = temp_image.rotate(rotation_angle, resample=Image.BICUBIC, expand=0)
    
    # Tạo ảnh cuối cùng
    final_image = Image.new('RGB', (width, height), 'white')
    
    # Chèn ảnh đã xoay vào ảnh cuối cùng, căn giữa
    paste_x = (width - (width + padding*2)) // 2
    paste_y = (height - (height + padding*2)) // 2
    final_image.paste(rotated_text, (paste_x, paste_y), rotated_text)
    
    draw = ImageDraw.Draw(final_image)
    
    # Thêm dấu cộng (+ được tạo bởi 5 dấu chấm)
    for _ in range(60):
        x = random.randint(1, width-2)
        y = random.randint(1, height-2)
        # Vẽ dấu cộng (+) bằng 5 điểm
        draw.point((x, y), fill=(0, 0, 0))      # Điểm trung tâm
        draw.point((x+1, y), fill=(0, 0, 0))    # Điểm bên phải
        draw.point((x-1, y), fill=(0, 0, 0))    # Điểm bên trái
        draw.point((x, y+1), fill=(0, 0, 0))    # Điểm bên dưới
        draw.point((x, y-1), fill=(0, 0, 0))    # Điểm bên trên
    
    # Thêm chấm đen mờ
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        # Sử dụng màu xám nhạt để tạo hiệu ứng mờ
        gray_level = random.randint(100, 180)
        draw.point((x, y), fill=(gray_level, gray_level, gray_level))

    # Thêm đường nhiễu
    for _ in range(12):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill='black', width=1)

    return final_image

# Sinh CAPTCHA
captcha_text = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
captcha_img = generate_centered_captcha(captcha_text)
captcha_img.save("captcha_times_centered.png")
