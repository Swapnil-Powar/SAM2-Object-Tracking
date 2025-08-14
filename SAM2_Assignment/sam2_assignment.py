import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil
import matplotlib.patches as patches
import requests
import zipfile

# SAM2 model configuration
checkpoint = "./sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# Initialize SAM2 models
try:
    predictor_prompt = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    sam2 = build_sam2(model_cfg, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("SAM2 models initialized successfully!")
except Exception as e:
    print(f"Error initializing SAM2: {e}")
    print("Please ensure SAM2 is installed and model files are available")

tempfolder = "./tempdir"

def create_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def cleardir(tempfolder):
    filepaths = glob.glob(tempfolder + "/*")
    for filepath in filepaths:
        os.unlink(filepath)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolors='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolors='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def process_img_png_mask(imgpath, maskpath, visualize=True):
    """
    Process image and mask to extract bounding box coordinates
    
    Args:
        imgpath: Path to the input image
        maskpath: Path to the ground truth mask
        visualize: Whether to display the results
    
    Returns:
        [xmin, xmax, ymin, ymax]: Bounding box coordinates
    """
    try:
        # Load image and mask
        image = Image.open(imgpath).convert('RGB')
        mask = Image.open(maskpath).convert('L')  # Convert to grayscale
        
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # Find non-zero pixels (object pixels)
        non_zero_coords = np.where(mask_array > 0)
        
        if len(non_zero_coords[0]) == 0:
            print("Warning: No object found in mask")
            return [0, 0, 0, 0]
        
        # Get bounding box coordinates
        ymin, ymax = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
        xmin, xmax = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
        
        if visualize:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Ground truth mask
            ax2.imshow(mask_array, cmap='gray')
            ax2.set_title('Ground Truth Mask')
            ax2.axis('off')
            
            # Image with bounding box
            ax3.imshow(image)
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)
            ax3.set_title('Extracted Bounding Box')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return [xmin, xmax, ymin, ymax]
        
    except Exception as e:
        print(f"Error processing image and mask: {e}")
        return [0, 0, 0, 0]

def track_item_boxes(imgpath1, imgpath2, img1boxclasslist, visualize=True):
    """
    Track objects between two images using SAM2 video predictor
    
    Args:
        imgpath1: Path to first image (where object is known)
        imgpath2: Path to second image (where object is to be tracked)
        img1boxclasslist: List of ([xmin,xmax,ymin,ymax], objectnumint) for all objects in imgpath1
        visualize: Whether to display results
    
    Returns:
        video_segments: Dictionary containing segmentation results
    """
    try:
        # imgpath1 :: Image where object is known
        # imgpath2 :: Image where object is to be tracked
        # img1boxclasslist :: [ ([xmin,xmax,ymin,ymax],objectnumint) ,....] for all objects in imagepath1
        create_if_not_exists(tempfolder)
        cleardir(tempfolder)
        shutil.copy(imgpath1, tempfolder + "/00000.jpg")
        shutil.copy(imgpath2, tempfolder + "/00001.jpg")
        inference_state = predictor_vid.init_state(video_path="./tempdir")
        predictor_vid.reset_state(inference_state)
        ann_frame_idx = 0
        for img1boxclass in img1boxclasslist:
            ([xmin, xmax, ymin, ymax], objectnumint) = img1boxclass
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor_vid.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=objectnumint,
                box=box,
            )
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        if visualize:
            fig, ax = plt.subplots()
            plt.title(f"original image object ::")
            ax.imshow(Image.open(tempfolder + "/00000.jpg"))
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
            out_frame_idx = 1
            plt.figure(figsize=(6, 4))
            plt.title(f"detected object in test image ::")
            plt.imshow(Image.open(tempfolder + "/00001.jpg"))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        return video_segments
    except Exception as e:
        print(f"Error in object tracking: {e}")
        return {}

def download_dataset():
    """Download CMU10_3D dataset if not already present"""
    dataset_url = "http://www.cs.cmu.edu/~ehsiao/3drecognition/CMU10_3D.zip"
    dataset_dir = "./CMU10_3D"
    
    if os.path.exists(dataset_dir):
        print("Dataset already exists")
        return True
    
    try:
        print("Downloading CMU10_3D dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open("CMU10_3D.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        with zipfile.ZipFile("CMU10_3D.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove("CMU10_3D.zip")
        print("Dataset downloaded and extracted successfully")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

# Main assignment execution as per the original code
if __name__ == "__main__":
    print("SAM2 Object Tracking Assignment")
    print("=" * 40)
    
    # Download dataset if needed
    download_dataset()
    
    # Assignment code as specified in the PDF
    firstimgpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000001.jpg'
    firstimgmaskpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000001_1_gt.png'
    
    # Process first image and mask
    [xmin, xmax, ymin, ymax] = process_img_png_mask(firstimgpath, firstimgmaskpath, visualize=True)
    
    # Second image path
    secondimgpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000002.jpg'
    secondimg = Image.open(secondimgpath)
    plt.imshow(secondimg)
    plt.show()
    
    # Track object between images
    op = track_item_boxes(firstimgpath, secondimgpath, [([xmin, xmax, ymin, ymax], 1)], visualize=True)
    
    # Extract results as per assignment
    output_masks = op[1]  # Mask for output image is always on op[1] for this example 
    print(output_masks)
    
    relevant_mask = output_masks[1]
    print(relevant_mask)
    
    print("Completed!")