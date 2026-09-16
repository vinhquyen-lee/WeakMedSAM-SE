import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import cv2
import json
from PIL import Image

# Không gian nhãn 9 loại u gốc của hệ thống BTRXD
TUMOR_CLASSES = [
    'osteochondroma', 'multiple osteochondromas', 'simple bone cyst', 
    'giant cell tumor', 'osteofibroma', 'synovial osteochondroma', 
    'other bt', 'osteosarcoma', 'other mt'
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TUMOR_CLASSES)}
PIC_SIZE = 256 

def smart_square_crop(image_np):
    """
    Tự động tìm vùng xương, tạo crop vuông, đệm nền đen chống méo ảnh,
    và trả về các thông số dịch chuyển để xử lý Mask.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Lọc nhiễu nhẹ và dùng Otsu để nhị phân hóa
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tìm vùng sáng
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:

        img_h, img_w = image_np.shape[:2]
        return image_np, 0, 0, max(img_w, img_h)
        
    # Lấy vùng sáng lớn nhất (phần xương/cơ thể chính)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Ép bounding box thành hình vuông
    center_x = x + w // 2
    center_y = y + h // 2
    side_length = max(w, h)
    
    side_length = int(side_length * 1.05)
    
    x1 = center_x - side_length // 2
    y1 = center_y - side_length // 2
    x2 = x1 + side_length
    y2 = y1 + side_length
    
    img_h, img_w, _ = image_np.shape
    
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)
    
    cropped_orig = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
    
    square_img = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    
    paste_x1 = crop_x1 - x1
    paste_y1 = crop_y1 - y1
    paste_x2 = paste_x1 + (crop_x2 - crop_x1)
    paste_y2 = paste_y1 + (crop_y2 - crop_y1)
    
    square_img[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_orig
    
    return square_img, x1, y1, side_length


def preprocess(args) -> None:
    img_path, json_path, output_dir, sample_name = args
    
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    
    square_img_np, offset_x, offset_y, side_length = smart_square_crop(img_np)
    
    img_square = Image.fromarray(square_img_np)
    img_resized = img_square.resize((PIC_SIZE, PIC_SIZE), Image.BILINEAR)
    img_resized.save(os.path.join(output_dir, f"{sample_name}.jpg"), quality=95)
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    scale = PIC_SIZE / side_length
    # print(f"DEBUG: {sample_name} | Offset: ({offset_x}, {offset_y}) | Scale: {scale}")
    
    mask = np.zeros((len(TUMOR_CLASSES), PIC_SIZE, PIC_SIZE), dtype=np.uint8)
    
    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue
        label = shape['label']
        if label not in CLASS_TO_IDX:
            continue
            
        channel_idx = CLASS_TO_IDX[label]

        orig_points = np.array(shape['points'], dtype=np.float32)
        scaled_points = np.empty_like(orig_points)
        
        scaled_points[:, 0] = (orig_points[:, 0] - max(0, offset_x)) * scale
        scaled_points[:, 1] = (orig_points[:, 1] - max(0, offset_y)) * scale

        scaled_points = np.clip(scaled_points, 0, PIC_SIZE - 1)

        # print(f"DEBUG: {sample_name} | Điểm mask đầu tiên sau scale: {scaled_points[0]}")
        
        scaled_points = np.round(scaled_points).astype(np.int32)
        
        cv2.fillPoly(mask[channel_idx], [scaled_points], 1)
        
    np.save(os.path.join(output_dir, f"{sample_name}_mask.npy"), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTRXD Preprocessing Pipeline (Smart Square Crop)")
    parser.add_argument("--input-path", type=str, help="Thư mục chứa file gốc", required=True)
    parser.add_argument("--output-path", type=str, help="Thư mục lưu kết quả", required=True)
    parser.add_argument("--workers", type=int, help="Số luồng xử lý song song", default=4)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    img_dict = {}
    json_dict = {}

    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            sample_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1].lower() 

            if ext in [".jpeg", ".jpg", ".png"]:
                img_dict[sample_name] = os.path.join(root, file)
            elif ext == ".json":
                json_dict[sample_name] = os.path.join(root, file)

    tasks = []
    for sample_name, img_path in img_dict.items():
        if sample_name in json_dict:
            json_path = json_dict[sample_name]
            tasks.append((img_path, json_path, args.output_path, sample_name))

    print(f"\n[INFO] Đã quét xong thư mục: {args.input_path}")
    print(f"[INFO] Tìm thấy tổng cộng {len(img_dict)} ảnh và {len(json_dict)} file JSON.")
    print(f"[INFO] Đã ghép thành công {len(tasks)} cặp ảnh-mask hợp lệ. Bắt đầu xử lý với {args.workers} luồng...\n")
    
    if len(tasks) > 0:
        with mp.Pool(args.workers) as pool:
            list(tqdm(pool.imap(preprocess, tasks), total=len(tasks), ncols=100))
    else:
        print("Không có tác vụ nào được chạy. Kiểm tra lại đường dẫn!")