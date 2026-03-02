import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("data/trained_model_10m.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("WHEAT DISEASE DETECTION SYSTEM")
    image_path = "home page2.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ### Welcome to the Wheat Disease Recognition System!!!
    
    My mission is to help in identifying some wheat diseases efficiently. Upload an image of a Wheat plant, and my system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** The system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose My System?
    - **Accuracy:** My system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Me
    Learn more about the project, dataset and some disease in the dataset on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                 This dataset was downloaded from kaggle. It was designed to empower researchers and developers in creating robust machine learning models for classifying various wheat plant diseases.
                It offers a collection of high-resolution images showcasing real-world wheat diseases without the use of artificial augmentation techniques.

                This dataset consists of about 14.2K rgb images of healthy and diseased crop leaves which are categorized into 15 different classes. 

                Some Of the diseases included in the dataset are:
                1. Brown Rust
                2. Yellow Rust
                3. Powdery Mildew
                4. Septoria Leaf Blotch
                5. Aphid [Pest]
                6. Mite [Pest]
                7. Stem Rust
                8. Common Root Rot etc.

                #### Content
                1. Train (13,104 images)
                2. Test (750 images)
                3. Validation (300 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 
               'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 
               'Mite', 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']
        st.success("The Model is Predicting it's a {}".format(class_name[result_index]))