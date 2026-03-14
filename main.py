import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os


# Custom CSS for pointer cursor
st.markdown("""
    <style>
        div[data-baseweb="select"] {
            cursor: pointer;
        }
        div[data-baseweb="select"] * {
            cursor: pointer !important;
        }
    </style>
""", unsafe_allow_html=True)



# Download model from Google Drive if not present
if not os.path.exists("trained_model_10m.keras"):
    gdown.download(
        "https://drive.google.com/uc?id=1RUkyD-2Wzp8LpxDLn8W9W9Y519VHIClW",
        "trained_model_10m.keras",
        quiet=False
    )

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model_10m.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction), prediction[0]  # return index AND all confidence scores

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if(app_mode == "Home"):
    st.header("WHEAT DISEASE DETECTION SYSTEM")
    image_path = "home page1.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    ### Welcome to the Wheat Disease Recognition System!!!
    
    My mission is to help in identifying some wheat diseases efficiently. Upload an image of a Wheat plant, and my system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** The system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results for further action.

    ### Why Choose My System?
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of my Wheat Disease Recognition System!

    ### About Project
    Learn more about the project, dataset and some disease in the dataset on the **About** page.
    """)

# About Project
elif(app_mode == "About"):
    st.header("About")
    st.markdown("""
    ### Dataset Information
    This dataset was downloaded from Kaggle. It was designed to empower researchers and developers 
    in creating robust machine learning models for classifying various wheat plant diseases.
    It offers a collection of high-resolution images showcasing real-world wheat diseases 
    without the use of artificial augmentation techniques.

    This dataset consists of about 14.2K RGB images of healthy and diseased crop leaves 
    which are categorized into 15 different classes.

    ---

    ### Diseases & Pests the Model Can Detect:

    | # | Disease/Pest | Type |
    |---|-------------|------|
    | 1 | Aphid | Pest |
    | 2 | Black Rust | Fungal Disease |
    | 3 | Blast | Fungal Disease |
    | 4 | Brown Rust | Fungal Disease |
    | 5 | Common Root Rot | Fungal Disease |
    | 6 | Fusarium Head Blight | Fungal Disease |
    | 7 | Healthy | Disease Free |
    | 8 | Leaf Blight | Fungal Disease |
    | 9 | Mildew | Fungal Disease |
    | 10 | Mite | Pest |
    | 11 | Septoria | Fungal Disease |
    | 12 | Smut | Fungal Disease |
    | 13 | Stem Fly | Pest |
    | 14 | Tan Spot | Fungal Disease |
    | 15 | Yellow Rust | Fungal Disease |

    ---

    ### Dataset Split
    | Split | Images |
    |-------|--------|
    | Train | 13,104 |
    | Test | 750 |
    | Validation | 300 |

    ---

    ### Model Performance
    - **Training Accuracy:** 88%
    - **Validation Accuracy:** 81%
    - **Total Classes:** 15
    - **Image Size:** 128 x 128 pixels
    """)

# Prediction Page
elif(app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        if test_image is not None:
            st.image(test_image, use_container_width=True)
        else:
            st.warning("⚠️ Please upload an image first!")
    
    # Predict button - now correctly indented inside Disease Recognition page
    if(st.button("Predict")):
        if test_image is not None:
            st.snow()
            st.write("Our Prediction")
            class_name = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot',
                          'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew',
                          'Mite', 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']

            result_index, confidence_scores = model_prediction(test_image)

            # Show predicted disease
            st.success("✅ The Model is Predicting it's a **{}**".format(class_name[result_index]))

            # Show confidence distribution
            st.write("### Confidence Scores:")
            for i, (disease, score) in enumerate(zip(class_name, confidence_scores)):
                percentage = float(score) * 100
                if i == result_index:
                    st.write(f"**🏆 {disease}**")
                else:
                    st.write(f"{disease}")
                st.progress(float(score), text=f"{percentage:.2f}%")
        else:
            st.warning("⚠️ Please upload an image first!")