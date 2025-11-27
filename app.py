import streamlit as st
import numpy as np
import os
import tempfile

# Configure TensorFlow before importing to avoid macOS threading issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Import PyArrow before TensorFlow to avoid mutex issues on macOS
try:
    import pyarrow
except ImportError:
    pass

# Delay TensorFlow import until needed
@st.cache_resource()
def get_tensorflow():
    import tensorflow as tf
    # Configure threading after import
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except:
        pass
    return tf

@st.cache_resource()
def load_model():
    tf = get_tensorflow()
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    return model

def model_prediction(test_image_path):
    tf = get_tensorflow()
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    
    model = load_model()
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x, verbose=0)
    return np.argmax(prediction)

##UI Part


st.title("Human Eye Disease Prediction")
st.markdown("""
### Human Eye Disease Prediction
This is a web app to predict the disease of the human eye.
""")

#Upload Image
test_image = st.file_uploader("Upload Image:")
if(test_image is not None):
    #Save to a temporary file
    temp_file_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as temp_file:
        temp_file.write(test_image.read())
        temp_file_path = temp_file.name
    # st.write(temp_file_path)
    st.image(test_image)

    #Predict
    if(st.button("Predict") and test_image is not None):
        with st.spinner("Predicting..."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

        st.success(f"The predicted disease is {class_name[result_index]}")

        # Recommendation
        with st.expander("Learn More"):
            #CNV
            if(result_index == 0):
                st.write("""
                CNV is a disease of the retina.
                """)
            #DME
            if(result_index == 1):
                st.write("""
                DME is a disease of the retina.
                """)
            #DRUSEN
            if(result_index == 2):
                st.write("""
                DRUSEN is a disease of the retina.
                """)
            #NORMAL
            if(result_index == 3):
                st.write("""
                NORMAL is a disease of the retina.
                """)

        #Clean up temp file
        os.remove(temp_file_path)


#Footer
st.markdown("""
By Godwins & Elikem
""")




