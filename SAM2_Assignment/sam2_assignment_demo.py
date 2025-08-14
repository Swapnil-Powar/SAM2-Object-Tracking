import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil
import matplotlib.patches as patches
from PIL import Image, ImageOps
import requests
import zipfile

# Configuration
tempfolder = "./tempdir"

# Mock SAM2 classes for demonstration
class MockSAM2VideoPredictor:
    def __init__(self):
        print("Mock SAM2 Video Predictor initialized (for demo purposes)")
    
    def init_state(self, video_path):
        print(f"Mock: Initializing state for video path: {video_path}")
        return {"video_path": video_path, "objects": {}}
    
    def reset_state(self, inference_state):
        print("Mock: Resetting inference state")
        inference_state["objects"] = {}
    
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        print(f"Mock: Adding box {box} for object {obj_id} at frame {frame_idx}")
        inference_state["objects"][obj_id] = {"box": box, "frame": frame_idx}
        return None, [obj_id], [np.ones((1, 100, 100))]
    
    def propagate_in_video(self, inference_state):
        print("Mock: Propagating objects in video")
        for frame_idx in [0, 1]:
            obj_ids = list(inference_state["objects"].keys())
            mock_masks = [np.random.random((1, 480, 640)) > 0.5 for _ in obj_ids]
            yield frame_idx, obj_ids, mock_masks

# Try to import SAM2, fall back to mock if not available
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2_video_predictor
    
    # Initialize real SAM2 models
    checkpoint = "./sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    
    predictor_prompt = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    sam2 = build_sam2(model_cfg, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    predictor_vid = build_sam2_video_predictor(model_cfg, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
    SAM2_AVAILABLE = True
    print("SAM2 successfully imported and initialized!")
    
except ImportError as e:
    print(f"SAM2 not available ({e}), using mock implementation for demo")
    predictor_vid = MockSAM2VideoPredictor()
    SAM2_AVAILABLE = False

def create_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def cleardir(tempfolder):
    filepaths = glob.glob(tempfolder + "/*")
    for filepath in filepaths:
        try:
            os.unlink(filepath)
        except:
            pass

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
    """
    try:
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
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor_vid.propagate_in_video(inference_state):
            if SAM2_AVAILABLE:
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            else:
                # Mock implementation
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
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
            plt.show()
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

def create_sample_data():
    """Create sample data if dataset is not available"""
    print("Creating sample data for demonstration...")
    
    sample_dir = "./sample_data"
    create_if_not_exists(sample_dir)
    
    # Create sample images
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img1[100:200, 150:250] = [255, 0, 0]  # Red rectangle
    img1_pil = Image.fromarray(img1)
    img1_pil.save(f"{sample_dir}/sample_001.jpg")
    
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2[110:210, 160:260] = [255, 0, 0]  # Slightly moved red rectangle
    img2_pil = Image.fromarray(img2)
    img2_pil.save(f"{sample_dir}/sample_002.jpg")
    
    # Create sample mask
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[100:200, 150:250] = 255  # White rectangle for mask
    mask_pil = Image.fromarray(mask)
    mask_pil.save(f"{sample_dir}/sample_001_mask.png")
    
    return (f"{sample_dir}/sample_001.jpg", 
            f"{sample_dir}/sample_001_mask.png", 
            f"{sample_dir}/sample_002.jpg")

# Main assignment execution
if __name__ == "__main__":
    print("SAM2 Object Tracking Assignment - Demo Version")
    print("=" * 50)
    print(f"SAM2 Status: {'Available' if SAM2_AVAILABLE else 'Mock Implementation'}")
    print()
    
    # Try to download dataset, fall back to sample data
    dataset_available = download_dataset()
    
    if dataset_available:
        firstimgpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000001.jpg'
        firstimgmaskpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000001_1_gt.png'
        secondimgpath = '.\\CMU10_3D\\CMU10_3D\\data_2D\\can_chowder_000002.jpg'
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [firstimgpath, firstimgmaskpath, secondimgpath]):
            print("Dataset files not found, using sample data...")
            firstimgpath, firstimgmaskpath, secondimgpath = create_sample_data()
    else:
        firstimgpath, firstimgmaskpath, secondimgpath = create_sample_data()
    
    # Assignment code as specified
    [xmin, xmax, ymin, ymax] = process_img_png_mask(firstimgpath, firstimgmaskpath, visualize=True)
    
    secondimg = Image.open(secondimgpath)
    plt.imshow(secondimg)
    plt.show()
    
    op = track_item_boxes(firstimgpath, secondimgpath, [([xmin, xmax, ymin, ymax], 1)], visualize=True)
    
    output_masks = op[1]  # Mask for output image is always on op[1] for this example 
    print(output_masks)
    
    relevant_mask = output_masks[1]
    print(relevant_mask)
    
    print("\nCompleted!")
    print(f"Implementation: {'Full SAM2' if SAM2_AVAILABLE else 'Demo with Mock SAM2'}")