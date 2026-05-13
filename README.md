# Camouflage Object Detection using Improved PraNet

## Overview
This project focuses on camouflage object detection using an improved PraNet architecture with:
- ResNet50 backbone
- SE Attention
- Reverse Attention
- Deep Supervision
- Hybrid Loss Functions

The model performs pixel-wise segmentation of hidden camouflaged objects.

---

## Dataset

Download COD10K dataset from:

https://dengpingfan.github.io/pages/COD.html

After downloading, place it inside:

data/
└── COD10K/
    ├── Train/
    │   ├── Image/
    │   ├── GT_Object/
    │
    ├── Test/
    │   ├── Image/
    │   ├── GT_Object/

---

## Architecture
- Encoder: ResNet50
- Decoder: Multi-scale feature fusion
- Attention: SE Block + Reverse Attention

---

## Metrics
- Mean IoU: 0.71
- Mean Dice Score: 0.78

---

## Features
- Camouflage segmentation
- Deep supervision
- Edge refinement
- Streamlit deployment

---

## Installation

```bash
pip install -r requirements.txt

## Training

```bash
python -m src.train