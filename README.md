# SAM2 Object Tracking Assignment  

## Overview
This assignment implements object tracking using SAM2 (Segment Anything Model 2) from Meta AI. The solution includes object segmentation from ground truth masks, bounding box extraction, and object tracking between consecutive frames.

## Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

## Installation

### 1. Install SAM2
```bash
# Clone the SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install SAM2
pip install -e .
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Assignment
```bash
python sam2_assignment.py
```

## Key Functions Implemented

### `process_img_png_mask(imgpath, maskpath, visualize=True)`
- **Purpose**: Extract bounding box coordinates from ground truth mask
- **Input**: Image path and corresponding mask path
- **Output**: [xmin, xmax, ymin, ymax] coordinates
- **Features**: Visualization of original image, mask, and bounding box

### `track_item_boxes(imgpath1, imgpath2, img1boxclasslist, visualize=True)`
- **Purpose**: Track objects between two consecutive frames using SAM2
- **Input**: Two image paths and bounding box information
- **Output**: Video segments with object masks
- **Features**: SAM2 video predictor for temporal consistency

## Dataset
The assignment uses the CMU10_3D dataset:
- **Source**: http://www.cs.cmu.edu/~ehsiao/3drecognition/CMU10_3D.zip
- **Auto-download**: The script automatically downloads and extracts the dataset
- **Images used**: 
  - `can_chowder_000001.jpg` (reference frame)
  - `can_chowder_000001_1_gt.png` (ground truth mask)
  - `can_chowder_000002.jpg` (target frame)

## Expected Output
The assignment will:
1. Download and extract the CMU10_3D dataset
2. Process the first image and ground truth mask
3. Extract bounding box coordinates
4. Display the second image
5. Track the object between frames
6. Show segmentation results

## Assignment Workflow
```python
# Extract bounding box from ground truth mask
[xmin, xmax, ymin, ymax] = process_img_png_mask(firstimgpath, firstimgmaskpath, visualize=True)

# Display second image
secondimg = Image.open(secondimgpath)
plt.imshow(secondimg)
plt.show()

# Track object between frames
op = track_item_boxes(firstimgpath, secondimgpath, [([xmin, xmax, ymin, ymax], 1)], visualize=True)

# Extract results
output_masks = op[1]  # Mask for output image is always on op[1]
relevant_mask = output_masks[1]
```

## Large Files Download
The following large files are not included in this repository due to size constraints.  
Please download them from Google Drive and place them in the project root before running the code.

1. **sam2_hiera_tiny.pt** – [Download Link]([https://drive.google.com/your_model_file_link](https://drive.google.com/file/d/1pr9IdNueI_RtZJePGnNEToBPVO3QYlAH/view?usp=sharing))
2. **CMU10_3D/data_2/core.l5239** – [Download Link]([https://drive.google.com/your_dataset_file_link](https://drive.google.com/drive/folders/1cFplc79RE0bnJETRBBq3AAbWdG-zbls8?usp=sharing))

## Technical Details
- **Framework**: PyTorch with SAM2
- **Visualization**: Matplotlib for displaying results
- **Image Processing**: PIL for image handling
- **Object Tracking**: SAM2 video predictor for temporal consistency

## Results
The assignment successfully demonstrates:
- Accurate bounding box extraction from ground truth masks
- Object tracking between consecutive frames
- Comprehensive visualization of all processing steps
- Integration with the CMU10_3D dataset

## References
- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
- CMU10_3D Dataset: http://www.cs.cmu.edu/~ehsiao/3drecognition/
