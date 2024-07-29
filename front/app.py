import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import uuid
import plotly.express as px
from PIL import Image

# Add the parent directory of 'back' to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'back'))
import app as backend

# Function to load and display the logo
def load_logo(logo_path):
    try:
        if logo_path.is_file():
            image = Image.open(logo_path)
            # Resize image to fit sidebar
            max_width = 190
            width_percent = (max_width / float(image.size[0]))
            new_height = int((float(image.size[1]) * float(width_percent)))
            resized_image = image.resize((max_width, new_height), Image.ANTIALIAS)
            st.sidebar.image(resized_image, use_column_width=False)
        else:
            st.sidebar.write("Logo not found. Please check the path.")
    except Exception as e:
        st.sidebar.write(f"Error loading logo: {e}")

# Try to read and display the logo image
logo_path = Path(r"C:\Users\Mohamed Razzegui\Desktop\!Dump\docs pfe\monitoring_system\logo.png")
load_logo(logo_path)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Audio Quality Evaluation", "Text Evaluation", "Summary of Results"])

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = pd.DataFrame()

if 'text_data' not in st.session_state:
    st.session_state.text_data = pd.DataFrame()

if 'file_path' not in st.session_state:
    st.session_state.file_path = None

if 'uploaded_csv' not in st.session_state:
    st.session_state.uploaded_csv = None

# Home Page
if page == "Home":
    st.title("Vocal Chatbot Monitoring System")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
      
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_dir = Path("./temporary_audios")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.file_path = file_path  # Save path in session state
        st.audio(uploaded_file)

    uploaded_csv = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_csv is not None:
        data = pd.read_csv(uploaded_csv)
        st.session_state.text_data = data  # Save CSV in session state
        st.session_state.uploaded_csv = uploaded_csv
        st.write("Uploaded CSV Data Preview:")
        st.dataframe(data, height=250, width=700)

# Audio Quality Evaluation Page
if page == "Audio Quality Evaluation":
    st.title("Audio Quality Evaluation")

    # Process the file and display results
    if st.button("Process Audio"):
        if 'file_path' in st.session_state:
            if st.session_state.file_path:
                base_name = Path(st.session_state.file_path).stem
                result_file = Path(f"./results/{base_name}/{base_name}_dataset.xlsx")

                # Check if the result file already exists
                if result_file.exists():
                    # Prompt the user with an option to overwrite or cancel processing
                    overwrite_option = st.radio("A file with the same name already exists. Do you want to overwrite it?", ("Yes, overwrite it", "No, cancel processing"))

                    if overwrite_option == "Yes, overwrite it":
                        snr, result_message = backend.process_audio_and_save_results(st.session_state.file_path)
                        st.success(result_message)

                        # Display the processed data
                        if result_file.exists():
                            df = pd.read_excel(result_file)
                            st.session_state.audio_data = df

                        # Update the mean SNR
                        if not st.session_state.audio_data.empty:
                            mean_snr = st.session_state.audio_data['SNR'].mean()
                            st.session_state.mean_snr = mean_snr
                    else:
                        st.warning("Processing cancelled.")
                else:
                    snr, result_message = backend.process_audio_and_save_results(st.session_state.file_path)
                    st.success(result_message)

                    # Display the processed data
                    if result_file.exists():
                        df = pd.read_excel(result_file)
                        st.session_state.audio_data = df

                    # Update the mean SNR
                    if not st.session_state.audio_data.empty:
                        mean_snr = st.session_state.audio_data['SNR'].mean()
                        st.session_state.mean_snr = mean_snr
            else:
                st.warning("Please upload an audio file first.")

    # Display the processed data
    if not st.session_state.audio_data.empty:
        st.dataframe(st.session_state.audio_data, height=250, width=700)

        # Calculate and display both mean SNRs
        mean_snr_all = st.session_state.audio_data['SNR'].mean()
        valid_audios = st.session_state.audio_data[st.session_state.audio_data['Valid Audio']]
        mean_snr_valid = valid_audios['SNR'].mean() if not valid_audios.empty else 'N/A'

        st.metric(label="Mean SNR (All Audios)", value=f"{mean_snr_all:.2f}")
        st.metric(label="Mean SNR (Valid Audios)", value=f"{mean_snr_valid:.2f}")

        # Get the audio file paths and their corresponding start and end times
        audio_files_with_time = [(audio_file, start_time, end_time) for audio_file, start_time, end_time in zip(st.session_state.audio_data['Audio File'], st.session_state.audio_data['Start Times'], st.session_state.audio_data['End Times'])]

        # Multiselect widget for selecting audio files with start and end times
        selected_audio_files = st.multiselect("Select audio files from the dataset", audio_files_with_time, format_func=lambda x: x[0])


        if st.button("Show Plots"):
            if selected_audio_files:
                waveform_container = st.container()
                recon_waveform_container = st.container()
                vad_container = st.container()
                signals_container = st.container()
                recon_signals_container = st.container()

                with waveform_container:
                    for audio_info in selected_audio_files:
                        audio_file, start_time, end_time = audio_info
                        fig = backend.plot_waveform(audio_file)
                        st.pyplot(fig)

                with recon_waveform_container:
                    for audio_info in selected_audio_files:
                        audio_file, start_time, end_time = audio_info
                        fig = backend.reconstructed_waveform(audio_file)
                        st.pyplot(fig)

                with vad_container:
                    for audio_info in selected_audio_files:
                        audio_file, start_time, end_time = audio_info
                        fig = backend.plot_vad(audio_file)
                        st.pyplot(fig)

                with signals_container:
                    for audio_info in selected_audio_files:
                        audio_file, start_time, end_time = audio_info
                        fig = backend.plot_signals(audio_file)
                        st.pyplot(fig)

                with recon_signals_container:
                    for audio_info in selected_audio_files:
                        audio_file, start_time, end_time = audio_info
                        fig = backend.recon_plot_signals(audio_file)
                        st.pyplot(fig)
                
            else:
                st.warning("Please select audio files from the dataset to plot their features.")

# Text Evaluation Page
if page == "Text Evaluation":
    st.title("Text Evaluation")

    
    # Process the data if the user clicks 'Evaluate Text'
    if st.button("Evaluate Text"):
        if not st.session_state.text_data.empty:
            
            # It should now return a DataFrame and two percentages
            text_data, satisfaction_rate, correctness_percentage = backend.process_text_data(st.session_state.text_data)
            st.session_state.text_data = text_data  # Save processed data in session state

            # Display processed data
            st.write("Processed Data:")
            st.dataframe(text_data, height=250, width=700)

            # Display key statistics using metrics
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.2f}%")
            st.metric("Correct Responses Percentage", f"{correctness_percentage:.2f}%")

            # Visualization
            st.write("Similarity Score Distribution")
            fig = px.histogram(text_data, x='similarity')
            st.plotly_chart(fig)

            st.write("Sentiment Analysis Results")
            sentiment_fig = px.histogram(text_data, x='sentiment')
            st.plotly_chart(sentiment_fig)
        else:
            st.warning("Please upload a csv file first.")
        

# Summary of Results Page
if page == "Summary of Results":
    st.title("Summary of Results")
    st.write("This section will summarize all results.")
