Introduction
This repository contains code for detecting gestures in images or short video clips using a pre-trained MobileNetV2 model and OpenCV. The model is trained to identify specific gestures and annotate them in real-time.

Requirements
Python 3.x
OpenCV
TensorFlow
numpy

Usage

Preparation:

Ensure you have the necessary dependencies installed.
Place your desired gesture representation (image or short video clip) in the repository directory.

Running the code:

Modify the paths to your input image or video file in the code.
Run the following command:
python gesture_detection.py
The annotated output will be saved as 'output_video.avi'.

Code Structure

gesture_detection.py: Main Python script containing the code for gesture detection.
requirements.txt: Text file listing all the required dependencies.
Parameters
confidence_threshold: Threshold for considering a gesture as detected.
font: Font type for annotation.
position: Position to display annotation on the frame.
font_scale: Font scale for annotation.
font_color: Color of the annotation text.
line_type: Line type for annotation.

Acknowledgments

This project utilizes the MobileNetV2 model pre-trained on ImageNet.
Thanks to the OpenCV and TensorFlow communities for their valuable contributions

Contact
For any inquiries or issues, please contact ashishchaudhary5252@gmail.com
