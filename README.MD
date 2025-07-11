# KITTI Dataset Resizer

A program for scaling images and annotations of the KITTI dataset to a uniform size of 284x284 pixels without cropping, with proportional recalculation of bounding box coordinates.

## Description

The script processes a dataset consisting of 200 images in JPG format and 200 annotation files in TXT format according to the KITTI standard. Each image is scaled to a size of 284x284 pixels using the OpenCV library. The corresponding bounding box coordinates in the annotations are recalculated proportionally based on the scaling factors.

The program performs an integrity check of the dataset before processing, ensuring that each JPG file has a corresponding TXT annotation file. All operations are logged to a file named resize.log and also printed to the console.

A graphical user interface based on Tkinter is provided, featuring a progress bar for visual monitoring of the processing. When launched with the nogui parameter, a textual progress bar is shown using the tqdm library.

A validation mode is implemented to verify the correctness of the processed results, including checks on image dimensions and the accuracy of bounding box coordinate recalculations.

## Requirements

The program requires Python version 3.13 or higher with the latest Tkinter version. The following libraries are needed:

opencv-python for image processing
tqdm for displaying progress in console mode
tkinter for the graphical interface (usually included in the standard Python distribution)

To install the required dependencies, run:
```
pip install opencv-python tqdm
```

## Input Data Structure
The input data must be organized into two folders:
The images folder contains 200 images in JPG format.
The kitti_annotations folder contains 200 annotation files in TXT format.
File names must match. For example:
image001.jpg corresponds to image001.txt.

## Output Data

The scaled images are saved in the images folder.
The recalculated annotations are saved in the annotations folder.
If the output folder is the same as the input folder, the program creates a backup of the original images in the images_orig folder, if necessary.

## How to run program

To launch the program with the graphical interface, use the command:
```
python main.py
```

To launch in console mode with a textual progress bar: 
```
python main.py --nogui
```

To run validation mode:
```
python main.py --mode validate
```

## Command-Line Parameters
The program supports the following command-line parameters:

mode – operation mode: resize for scaling or validate for result verification (default: resize)

width and height – target image dimensions in pixels (default: 284)

input-images – folder with the original images (default: images)

input-annots – folder with the original annotations (default: kitti_annotations)

output-images – folder to save scaled images (default: images)

output-annots – folder to save scaled annotations (default: annotations)

nogui – launch without graphical interface

Additional parameters for validation mode:
orig-images – folder with the original images (default: images_orig)

orig-annots – folder with the original annotations (default: kitti_annotations)

resized-images – folder with the scaled images (default: images)

resized-annots – folder with the scaled annotations (default: annotations)

tolerance – allowed error margin when validating coordinates (in pixels) (default: 1.0)

## Usage Examples
Standard processing with graphical interface:
```
python main.py
```

Processing in console mode:
```
python main.py --nogui
```

Scaling to size 512x512:
```
python main.py --width 512 --height 512
```

Processing with custom input/output paths:
```
python main.py --input-images ./dataset/images --input-annots ./dataset/labels --output-images ./results/images --output-annots ./results/labels
```

Validation of processing results::
```
python main.py --mode validate
```

## KITTI Annotation Format

The program works with annotations in the KITTI format, where each line contains information about one object. Only the bounding box coordinates located at positions 4–7 are processed: bbox_left, bbox_top, bbox_right, bbox_bottom. The other fields remain unchanged.

## Logging

All operations are recorded in the file resize.log and output to the console. Detailed statistics are maintained for each processed file, including information about image dimensions, scaling factors, and the number of objects processed in the annotations.

## Validation

The validation mode verifies the correctness of the processed results by comparing image sizes and bounding box coordinates. For each object, it checks that the coordinates are scaled with the correct factors within the specified tolerance.

## Program Architecture

All functionality is contained in a single Python file, main.py. The program consists of functions for image processing, annotation processing, dataset integrity checking, and result validation. The graphical user interface is implemented as a separate class with multithreading support to prevent the interface from freezing during processing.