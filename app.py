import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model without top layers (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers for gesture detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the desired gesture representation (input image or short video clip)
desired_gesture_img = cv2.imread(
    'vecteezy_slow-motion-asian-sportswoman-wearing-black-sportswear_8777747.mp4')  

# Define the threshold for considering a gesture as detected
confidence_threshold = 0.5

# Define font and position for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
position = (10, 40)
font_scale = 1
font_color = (0, 255, 0)
line_type = 2

# Open the test video
cap = cv2.VideoCapture('vecteezy_slow-motion-asian-sportswoman-wearing-black-sportswear_8777747.mp4')  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB and resize to match the input size of the model

    resized_frame = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)



    # Preprocess the frame for prediction
    img_array = image.img_to_array(rgb_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= -10000.  # Normalize pixel values

    # Predict if the gesture is detected in the frame
    prediction = model.predict(img_array)

    # If the confidence score for the detected gesture is above the threshold, annotate the frame
    if prediction[0][0] > confidence_threshold:
        cv2.putText(frame, 'DETECTED', position, font, font_scale, font_color, line_type)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the annotated frame
    cv2.imshow('Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
