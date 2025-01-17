import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from PIL import Image
import logging
import traceback
from pathlib import Path
from torchvision.ops import box_convert
from Grounded_SAM_2.sam2.build_sam import build_sam2
from Grounded_SAM_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from Grounded_SAM_2.grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from recognize_anything.ram.models import ram_plus
from recognize_anything.ram import inference_ram as inference
from recognize_anything.ram import get_transform
from supervision import Color, ColorPalette, ColorLookup

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).parent / "Grounded_SAM_2"))
sys.path.append(str(Path(__file__).parent / "recognize_anything"))


"Use absolute path"
SAM2_CHECKPOINT = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
#DEVICE = "cuda" if torch.cuda.is_available() else 
DEVICE = "cpu"
DUMP_JSON_RESULTS = True
IMAGE_SIZE = 384

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--source-folder',
                    metavar='DIR',
                    help='path to dataset',
                    default='')
parser.add_argument('--target-folder',
                    metavar='DIR',
                    help='path to output folder',
                    default='')
parser.add_argument('--kernel-size',
                    type=int,           
                    default=7,          
                    help='Size of median kernel')
parser.add_argument('--min-threshold',
                    type=int,           
                    default=5,          
                    help='min threshold canny edge')
parser.add_argument('--max-threshold',
                    type=int,           
                    default=10,          
                    help='max threshold canny edge')

args = parser.parse_args()
IMAGE_DIR = args.source_folder
OUTPUT_DIR = Path(args.target_folder)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "segment").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "edge").mkdir(parents=True, exist_ok=True)

transform = get_transform(image_size=IMAGE_SIZE)
#Checkpoint of RAM
model_ram = ram_plus(pretrained='C:/Users/ADMIN/Downloads/ram_plus_swin_large_14m.pth', image_size=IMAGE_SIZE, vit='swin_l')
model_ram.eval()
model_ram = model_ram.to(DEVICE)

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def edge_detection(image_path, output_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or unable to read.")

        edges = cv2.Canny(image, 20, 50)

        cv2.imwrite(output_path, edges)

    except Exception as e:
        error_message = f"Error processing {image_path}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        print(error_message)

def process_image(image_path, args):
    image = transform(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    res = inference(image, model_ram)
    class_recoginzed = [tag.strip() for tag in res[0].split('|')]
    additional_class = ["hair", "eye", "nose", "lips", "teeth", "ear", "glasses", "face", "mouth", "eyebrow"]
    for item in additional_class:
        if item not in class_recoginzed:
            class_recoginzed += [item]
    text_prompt = ""
    for item in class_recoginzed:
        text_prompt += item + ". "
    text_prompt = text_prompt[:-1]

    
    image_source, image_data = load_image(image_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_data,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    scores = scores.squeeze().tolist() 
    
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
   
    img = cv2.imread(image_path)
    
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    ) 

    if img is not None:
        img[:] = 0
    
    num_classes = len(set(detections.class_id))
    palette = create_color_palette(num_classes)
    mask_annotator = sv.MaskAnnotator(palette)
    
    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)


    annotated_frame = cv2.medianBlur(annotated_frame, args.kernel_size)

    output_extension = Path(image_path).suffix 
    output_mask_path = os.path.join(OUTPUT_DIR, f"segment/{Path(image_path).stem}_segment{output_extension}")
    cv2.imwrite(output_mask_path, annotated_frame)

    output_edge_path = os.path.join(OUTPUT_DIR, f"edge/{Path(image_path).stem}_edge{output_extension}")
    edge_detection(output_mask_path, output_edge_path)

    
    
def create_color_palette(num_colors: int) -> ColorPalette:
    cmap = plt.cm.get_cmap('tab20', num_colors) 
    colors = []
    for i in range(num_colors):
        r, g, b, _ = cmap(i)
        color = Color(
            r=int(255 * r),
            g=int(255 * g),
            b=int(255 * b)
        )
        colors.append(color)
    return ColorPalette(colors=colors)

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file_name in files:
            if not file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            image_path = os.path.join(root, file_name)

            logging.basicConfig(filename="errors.log", level=logging.ERROR)

            try:
                process_image(image_path, args)
                print(f"Processed: {image_path}")
            except Exception as e:
                error_message = f"Error processing {image_path}: {e}"
                print(error_message)
                logging.error(error_message)
                logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()