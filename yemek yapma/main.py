import streamlit as st
import tensorflow as tf
import numpy as np
from gtts import gTTS
import openai
import tempfile
import os
import base64

# OpenAI API anahtarını buraya eklendi
openai.api_key = 'your API key'

# Function to generate and save audio
def generate_audio(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_audio_file.name)
            temp_audio_file_path = temp_audio_file.name

        # Ensure the file is closed before using it
        temp_audio_file.close()

        # Ensure the file exists before trying to play it
        if os.path.exists(temp_audio_file_path):
            audio_file = open(temp_audio_file_path, 'rb')
            audio_bytes = audio_file.read()
            audio_file.close()
            return audio_bytes, temp_audio_file_path

        else:
            st.error("Audio file was not created successfully.")
            return None, None
    except Exception as e:
        st.error(f"An error occurred while generating audio: {e}")
        return None, None

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Session state to keep track of results
if 'results' not in st.session_state:
    st.session_state.results = []

if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES MASTER")
    image_path = "home_img.jpg"
    st.image(image_path, use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Us")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code(
        "vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Fruits&Vegetables Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Predict button
    if test_image and st.button("Predict"):

        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = [i.strip() for i in content]
        prediction_text = f"it's a {label[result_index]}"
        st.success(prediction_text)
        st.session_state.results.append(label[result_index])

        # Generate audio of the prediction
        audio_bytes, audio_path = generate_audio(prediction_text)
        if audio_bytes:
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_path = audio_path

        # Display audio player and play audio automatically
        if st.session_state.audio_bytes:
            audio_str = "data:audio/mp3;base64," + base64.b64encode(st.session_state.audio_bytes).decode()
            audio_html = f"""
                <audio id="audio_player" autoplay>
                <source src="{audio_str}" type="audio/mp3">
                Your browser does not support the audio element.
                </audio>
                <script>
                var audio = document.getElementById('audio_player');
                audio.play();
                </script>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

    # Finish button
    if st.button("Finish"):
        if not st.session_state.results:
            st.warning("No predictions to show.")
        else:
            st.write("All Predictions:")
            st.write(st.session_state.results)

            # OpenAI API ile ChatGPT'ye malzemelerle yemek önerisi sorma
            prompt = f"These are the ingredients I have: {', '.join(st.session_state.results)}. Can you suggest a recipe using these ingredients and provide its approximate calorie count?"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )

            # ChatGPT'den gelen öneriyi ekrana yazdırma
            recipe = response['choices'][0]['message']['content']
            st.write("Recommended Recipe and Calorie Information:")
            st.write(recipe)

            # Generate audio of the recipe and calorie information
            audio_bytes, audio_path = generate_audio(recipe)
            if audio_bytes:
                st.session_state.audio_bytes = audio_bytes
                st.session_state.audio_path = audio_path

                # Display audio player and play audio automatically
                audio_str = "data:audio/mp3;base64," + base64.b64encode(st.session_state.audio_bytes).decode()
                audio_html = f"""
                    <audio id="audio_player" autoplay>
                    <source src="{audio_str}" type="audio/mp3">
                    Your browser does not support the audio element.
                    </audio>
                    <script>
                    var audio = document.getElementById('audio_player');
                    audio.play();
                    </script>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

        # Sonuçları temizlemek için listeyi sıfırlama
        st.session_state.results = []
