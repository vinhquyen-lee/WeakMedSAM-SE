# WeakMedSAM-SE

> **Phân đoạn ảnh X-quang xương dựa trên mô hình học giám sát yếu**  
> *Bone X-ray Image Segmentation Based on Weakly Supervised Learning*

Nghiên cứu bài toán **phân đoạn tổn thương xương trên ảnh X-quang trong điều kiện chỉ sử dụng nhãn yếu ở cấp độ ảnh**.

Nghiên cứu được xây dựng và kế thừa từ khung [**WeakMedSAM**](https://arxiv.org/pdf/2503.04106), kết hợp các mô-đun **Sub-Class Exploration**, **Prompt Affinity Mining** và **Random Walk**. Đồng thời, nghiên cứu đề xuất chiến lược **lựa chọn điểm gợi ý động dựa trên Shannon Entropy**, nhằm tận dụng tốt hơn thông tin không chắc chắn trên bản đồ kích hoạt và hỗ trợ mô hình **SAM** tạo nhãn giả chất lượng cao hơn.

---

## Dataset

Nghiên cứu sử dụng các bộ dữ liệu:

- [**BTXRD — Bone Tumor X-ray Radiograph Dataset**](https://www.kaggle.com/datasets/bhavyasahu/btrxd-original-1)

- [**BraTS 2019 Dataset**](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

### Experimental Environment

Các thực nghiệm được triển khai trên **Kaggle Notebook** với GPU hỗ trợ CUDA.

---

## Project Structure

```text
WeakMedSAM-SE/
│
├── brats/
│   ├── preprocess.py
│   └── dataset.py
│
├── btxrd/
│   ├── btxrd_preprocess.py
│   └── dataset.py
│
├── samus/
│   ├── modeling/
│   ├── utils/
│   ├── __init__.py
│   ├── automatic_mask_generator.py
│   ├── build_sam_us.py
│   └── SamPredictor.py
│
├── unet/
│   ├── __init__.py
│   ├── unet_model.py
│   └── unet_parts.py
│
├── utils/
│   ├── affinity.py
│   ├── metrics.py
│   ├── pytutils.py
│   └── torchutils.py
│
├── cluster.py
├── train.py
├── lab_gen.py
├── train_unet.py
├── eval.py
│
└── README.md
```
---

### Data Preparing
Tiền xử lý bộ dữ liệu:
```bash
python btxrd/btxrd_preprocess.py \
    --input-path /kaggle/input/datasets/bhavyasahu/btrxd-original-1/BTXRD \
    --output-path ./btrxd_out \
    --workers 4
```

### Pre-clustering
Thực hiện phân cụm đặc trưng để tạo các sub-classes:
```bash
python cluster.py \
    --data_path ./btrxd_out \
    --save_path ./btrxd_out \
    --data_module btxrd \
    --batch_size 64 \
    --parent_classes 9 \
    --child_classes 4 \
    --gpus 0
```

### Train WeakMedSAM

Download the checkpoint of SAM ViT-b from metaAI:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Training:

```bash
python train.py \
    --seed 42 \
    --sam_ckpt ./sam_vit_b_01ec64.pth \
    --lr 1e-4 \
    --batch_size 8 \
    --max_epochs 20 \
    --val_iters 100 \
    --index btrxd_weakmedsam_entropy \
    --data_path ./btrxd_out \
    --data_module btxrd \
    --parent_classes 1 \
    --child_classes 36 \
    --child_weight 0.5 \
    --cluster_file ./btrxd_out/btrxd-4.bin \
    --logdir ./logs \
    --gpus 0
```

### Generating Pseudo Labels
Sử dụng mô hình WeakMedSAM đã huấn luyện để sinh pseudo labels:
```bash
python lab_gen.py \
    --batch-size 8 \
    --data-path ./btrxd_out \
    --save-path ./btrxd_pseudo_labels \
    --data-module btxrd \
    --parent-classes 1 \
    --child-classes 36 \
    --samus-ckpt ./logs/btrxd_weakmedsam_entropy/btrxd_weakmedsam_entropy_latest.pth \
    --sam-ckpt ./sam_vit_b_01ec64.pth \
    --t 4 \
    --beta 8 \
    --threshold 0.5 \
    --gpus 0
```

### Training Segmentation Network
Huấn luyện mạng U-Net sử dụng các pseudo labels đã sinh:
```bash
mkdir -p tblog/btrxd_unet

python train_unet.py \
    --seed 42 \
    --lr 1e-4 \
    --batch_size 16 \
    --max_epochs 50 \
    --val_iters 100 \
    --index btrxd_unet \
    --data_path ./btrxd_out \
    --lab_path ./btrxd_pseudo_labels \
    --data_module btxrd \
    --num_classes 10 \
    --logdir ./tblog \
    --gpus 0
```

### Evaluation
Mô hình được đánh giá bằng các độ đo sau:

| Metric | Mục đích |
|---|---|
| **Dice Coefficient** | Đánh giá mức độ chồng lấp giữa hai vùng phân đoạn |
| **IoU (Jaccard Index)** | Đánh giá mức độ giao nhau giữa vùng dự đoán và vùng tham chiếu |
| **ASSD** | Đánh giá khoảng cách bề mặt đối xứng trung bình |
| **HD95** | Đánh giá sai lệch biên tại phân vị 95% |

```bash
python eval.py \
    --data_path ./btrxd_out \
    --data_module btxrd \
    --batch_size 16 \
    --num_classes 10 \
    --ckpt ./tblog/btrxd_unet/btrxd_unet.pth \
    --gpus 0
```

---

