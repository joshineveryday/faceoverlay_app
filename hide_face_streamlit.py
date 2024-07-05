import cv2
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the DNN model.
modelFile = 'model/res10_300x300_ssd_iter_140000.caffemodel'
configFile = 'model/deploy.prototxt'

# Read the model and create a network object.
net = cv2.dnn.readNetFromCaffe(prototxt=configFile, caffeModel=modelFile)

def face_overlay_rect(src_image, overlay_image, net, detection_threshold=0.9):
    img = src_image.copy()[:, :, :3]

    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    # Pass the blob to the DNN model.
    net.setInput(blob)
    # Retrieve detections from the DNN model.
    detections = net.forward()

    (h, w) = img.shape[:2]

    # Process the detections.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            overlay_image_resized = cv2.resize(overlay_image[:, :, :3], (int(x2) - int(x1), int(y2) - int(y1)))
            img[y1:y2, x1:x2] = overlay_image_resized

    return img

#### UI Interface
st.title("Image Overlay On Faces")

#Source Image Upload
source_image_buffer = st.file_uploader("Choose an image to upload", type=['jpg', 'jpeg', 'png'])

#Overlay Image Upload
overlay_image_buffer = st.file_uploader("Upload a face to overlay on the image", type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)

with col1:
    st.subheader('Source Image:')
    if source_image_buffer is not None:
        # Read the file and convert it to opencv Image.
        image = np.array(Image.open(source_image_buffer))
        st.image(image)

with col2:
    st.subheader('Overlay Image:')
    if overlay_image_buffer is not None:
        # Read the file and convert it to opencv Image.
        image = np.array(Image.open(overlay_image_buffer))
        st.image(image)

st.subheader('Result')
if source_image_buffer is not None and overlay_image_buffer is not None:
    # Read the file and convert it to opencv Image.
    image = np.array(Image.open(source_image_buffer))
    overlay_image = np.array(Image.open(overlay_image_buffer))
    img_result = face_overlay_rect(image, overlay_image, net)
    st.image(img_result)

