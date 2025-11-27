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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">Human Eye Disease Prediction</h1>', unsafe_allow_html=True)

# Project Description
with st.container():
    st.markdown("""
    <p>
    This AI-powered web application uses deep learning to analyze retinal images and predict 
    potential eye diseases. Simply upload a retinal scan image, and our trained model will 
    classify it into one of four categories: CNV (Choroidal Neovascularization), DME (Diabetic 
    Macular Edema), DRUSEN, or NORMAL.
    </p>
    <p><strong>How it works:</strong> Upload a retinal image below, click predict, and receive 
    an instant diagnosis with detailed information about the detected condition.</p>
    """, unsafe_allow_html=True)

st.divider()

# Upload Section
st.subheader("Upload Retinal Image")
st.markdown("Please upload a retinal scan image for analysis (supported formats: JPG, PNG, etc.)")

test_image = st.file_uploader("Upload an image file")

if test_image is not None:
    # Save to a temporary file
    temp_file_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as temp_file:
        temp_file.write(test_image.read())
        temp_file_path = temp_file.name
    
    # Display image in a container
    st.image(test_image, caption="Uploaded Retinal Image")
    
    # Predict Button
    predict_button = st.button("Predict Disease")
    
    if predict_button and test_image is not None:
        with st.spinner("Analyzing image... This may take a few seconds."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            disease_names = {
                'CNV': 'Choroidal Neovascularization',
                'DME': 'Diabetic Macular Edema',
                'DRUSEN': 'Drusen',
                'NORMAL': 'Normal (No Disease Detected)'
            }
        
        # Result Display
        result_color = {'CNV': 'orange', 'DME': 'orange', 'DRUSEN': 'orange', 'NORMAL': 'green'}
        
        st.markdown(f"""
        <div class="prediction-result" style="color: {result_color[class_name[result_index]]};">
            <strong>Prediction: {class_name[result_index]}</strong><br>
            <small>{disease_names[class_name[result_index]]}</small>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Information
        with st.expander("Learn More About This Condition", expanded=False):
            # CNV
            if result_index == 0:
                st.markdown("""
                ### CNV (Choroidal Neovascularization)
                **Choroidal Neovascularization** is an abnormal growth of blood vessels from the 
                choroid layer of the eye into the retina. This condition is commonly associated 
                with age-related macular degeneration (AMD) and can lead to vision loss if left 
                untreated.
                
                **Recommendation:** Consult with an ophthalmologist for a comprehensive eye examination 
                and discuss treatment options such as anti-VEGF injections or photodynamic therapy.
                """)
            # DME
            elif result_index == 1:
                st.markdown("""
                ### DME (Diabetic Macular Edema)
                **Diabetic Macular Edema** is a complication of diabetic retinopathy where fluid 
                accumulates in the macula, the central part of the retina responsible for sharp vision. 
                This condition can cause blurred vision and is a leading cause of vision loss in 
                people with diabetes.
                
                **Recommendation:** It's important to manage blood sugar levels and consult with a 
                retina specialist. Treatment options may include anti-VEGF injections, laser therapy, 
                or corticosteroids.
                """)
            # DRUSEN
            elif result_index == 2:
                st.markdown("""
                ### DRUSEN
                **Drusen** are yellow deposits under the retina that are often an early sign of 
                age-related macular degeneration (AMD). While small drusen may not affect vision, 
                larger drusen can increase the risk of developing advanced AMD.
                
                **Recommendation:** Regular monitoring with an eye care professional is recommended. 
                Lifestyle changes such as a healthy diet rich in antioxidants and regular eye 
                examinations can help manage the condition.
                """)
            # NORMAL
            elif result_index == 3:
                st.markdown("""
                ### NORMAL (No Disease Detected)
                **Great news!** The retinal image appears normal with no signs of the common eye 
                diseases we screen for (CNV, DME, or DRUSEN).
                
                **Recommendation:** Continue with regular eye examinations as recommended by your 
                eye care professional. Early detection is key to maintaining healthy vision.
                """)

        # Clean up temp file
        os.remove(temp_file_path)

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>Developed by <strong>Godwins & Elikem</strong></p>
    <p style="font-size: 0.9rem;">AI-Powered Eye Disease Detection System</p>
</div>
""", unsafe_allow_html=True)




