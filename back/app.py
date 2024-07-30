import os
import numpy as np
import pandas as pd
import torchaudio
import scipy
from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import matplotlib.pyplot as plt
from pyannote.core import SlidingWindowFeature, SlidingWindow
from pathlib import Path

#####################  Audio

# Instantiate the model
model = Model.from_pretrained(
    "pyannote/segmentation",
    use_auth_token="**********************************" #type your own token
)

# Function to calculate power of a signal
def calculate_power(signal):
    return torch.sum(signal ** 2) / signal.numel()

# Function to calculate SNR
def calculate_snr(signal, noise):
    power_signal = calculate_power(signal)
    power_noise = calculate_power(noise)
    return 10 * torch.log10(1+(power_signal / power_noise))

def get_vad_intervals(vad, start_time=0.0, end_time=None):
    intervals = []
    total_duration = 0.0  # Initialize total VAD duration
    for segment in vad.get_timeline():
        if segment.end > start_time and (end_time is None or segment.start < end_time):
            interval_start = max(segment.start, start_time)
            interval_end = min(segment.end, end_time) if end_time is not None else segment.end
            if interval_start != interval_end:  # Ensure we don't capture zero-length intervals
                intervals.append(f"{interval_start:.2f} --> {interval_end:.2f}")
                total_duration += interval_end - interval_start
    return ', '.join(intervals), total_duration

def save_audio_segment(audio_segment, sample_rate, output_path):
    torchaudio.save(output_path, audio_segment, sample_rate)

def split_audio_on_silence(audio, vad, sample_rate, min_silence_duration=3.0):
    segments = []
    current_start = None
    for segment in vad.get_timeline():
        if current_start is None:
            current_start = segment.start
        if segment.end - current_start >= min_silence_duration:
            segments.append(Segment(current_start, segment.end))
            current_start = segment.end

    # Process any remaining audio if not ended by silence
    if current_start is not None and current_start < audio.shape[1] / sample_rate:
        segments.append(Segment(current_start, audio.shape[1] / sample_rate))

    audio_segments = []
    for segment in segments:
        start_frame = int(segment.start * sample_rate)
        end_frame = int(segment.end * sample_rate)
        audio_segments.append(audio[:, start_frame:end_frame])

    return segments, audio_segments

def process_audio_and_save_results(audio_file_path, result_dir='./results'):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    audio_duration = waveform.shape[1] / sample_rate

    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.0, "min_duration_off": 0.0}
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_file_path)

    # Get the base name of the audio file
    base_name = Path(audio_file_path).stem

    # Create a directory for the conversation if it doesn't exist
    audio_dir = Path(result_dir) / base_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Split the audio based on silence
    segments, audio_segments = split_audio_on_silence(waveform, vad, sample_rate)

    results = []
    for i, (segment, audio_segment) in enumerate(zip(segments, audio_segments)):
        segment_path = audio_dir / f"segment_{i+1}.wav"
        save_audio_segment(audio_segment, sample_rate, segment_path)

        # Process the segment
        output = model(audio_segment.unsqueeze(1))
        _output = output.detach()[0].numpy()

        # Assuming _output has two columns: first for signal probabilities and second for noise probabilities
        signal_probs = _output[:, 0]
        noise1_probs = _output[:, 1]
        noise2_probs = _output[:, 2]

        # Interpolate probabilities using spline interpolation to match the length of the waveform
        interpolated_signal_probs = scipy.interpolate.interp1d(np.arange(len(signal_probs)), signal_probs, kind='cubic')(np.linspace(0, len(signal_probs) - 1, waveform.shape[1]))
        interpolated_noise1_probs = scipy.interpolate.interp1d(np.arange(len(noise1_probs)), noise1_probs, kind='cubic')(np.linspace(0, len(noise1_probs) - 1, waveform.shape[1]))
        interpolated_noise2_probs = scipy.interpolate.interp1d(np.arange(len(noise2_probs)), noise2_probs, kind='cubic')(np.linspace(0, len(noise2_probs) - 1, waveform.shape[1]))

        # Convert interpolated probabilities to PyTorch tensors
        signal_probs_tensor = torch.from_numpy(interpolated_signal_probs)
        noise_probs1_tensor = torch.from_numpy(interpolated_noise1_probs)
        noise_probs2_tensor = torch.from_numpy(interpolated_noise2_probs)

        # Reconstruct the signal and noise
        signal_audio = waveform[0] * signal_probs_tensor
        noise1_audio = waveform[0] * noise_probs1_tensor
        noise2_audio = waveform[0] * noise_probs2_tensor

        # Calculate SNR
        snr = calculate_snr(signal_audio, noise1_audio + noise2_audio)

        # Get VAD intervals specific to this segment
        segment_vad_intervals, vad_duration = get_vad_intervals(vad, start_time=segment.start, end_time=segment.end)

        # Split the VAD intervals by commas
        vad_intervals = segment_vad_intervals.split(',')

        # Extract the start and end times of the first interval if it's not empty
        if vad_intervals[0].strip():
            first_interval_start, first_interval_end = map(float, vad_intervals[0].strip().split('-->'))
        else:
            first_interval_start, first_interval_end = None, None

        # Extract the start and end times of the last interval if it's not empty
        if vad_intervals[-1].strip():
            last_interval_start, last_interval_end = map(float, vad_intervals[-1].strip().split('-->'))
        else:
            last_interval_start, last_interval_end = None, None


        # Determine if this segment is "valid" based on the VAD duration
        is_valid = vad_duration > 0.5

        segment_data = {
            'Audio File': str(segment_path),
            'SNR': snr.item(),
            'VAD Intervals': segment_vad_intervals,
            'Start Times': first_interval_start,
            'End Times': last_interval_end,
            'Valid Audio': is_valid 
        }
        results.append(segment_data)
    

    # Save results to an Excel file specific to the long audio
    print("result_dir:", result_dir)
    print("base_name:", base_name)
    result_file = audio_dir / f"{base_name}_dataset.xlsx"

    new_df = pd.DataFrame(results)
    new_df.to_excel(result_file, index=False)

    return snr, f"Results for {audio_file_path} have been saved to '{result_file}'"

def plot_waveform(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    _waveform = waveform.numpy().T
    time_axis = np.linspace(0, len(_waveform) / sample_rate, num=len(_waveform), endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_axis, waveform[0].numpy())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.grid(True)

    return fig



def reconstructed_waveform(audio_file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Calculate the time values for the x-axis
    time_values = np.arange(waveform.shape[1]) / sample_rate
    
    # Calculate the duration of the audio file in seconds
    audio_duration = waveform.shape[1] / sample_rate

    # Create a Segment object that spans the entire duration of the audio
    SAMPLE_CHUNK = Segment(0, audio_duration)

    # Voice Activity Detection pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_file_path)

    # Apply the model on the waveform
    output = model(waveform.unsqueeze(1))

    # Detach the output from the computation graph to avoid backpropagation
    _output = output.detach()[0].numpy()

    # Obtain the size of the output window
    window_size = _output.shape[0]

    # Calculate the duration of frames in seconds based on the output window size
    frame_duration = SAMPLE_CHUNK.duration / window_size

    # Create a time axis in seconds for the output frames
    time_axis_output = np.linspace(start=SAMPLE_CHUNK.start, stop=SAMPLE_CHUNK.start + SAMPLE_CHUNK.duration, num=window_size, endpoint=False)

    # Assuming _output has two columns: first for signal probabilities and second for noise probabilities
    signal_probs = _output[:, 0]
    noise1_probs = _output[:, 1]
    noise2_probs = _output[:, 2]

    # Interpolate probabilities using spline interpolation to match the length of the waveform
    interpolated_signal_probs = scipy.interpolate.interp1d(np.arange(len(signal_probs)), signal_probs, kind='cubic')(np.linspace(0, len(signal_probs) - 1, waveform.shape[1]))
    interpolated_noise1_probs = scipy.interpolate.interp1d(np.arange(len(noise1_probs)), noise1_probs, kind='cubic')(np.linspace(0, len(noise1_probs) - 1, waveform.shape[1]))
    interpolated_noise2_probs = scipy.interpolate.interp1d(np.arange(len(noise2_probs)), noise2_probs, kind='cubic')(np.linspace(0, len(noise2_probs) - 1, waveform.shape[1]))

    # Convert interpolated probabilities to PyTorch tensors
    signal_probs_tensor = torch.from_numpy(interpolated_signal_probs)
    noise1_probs_tensor = torch.from_numpy(interpolated_noise1_probs)
    noise2_probs_tensor = torch.from_numpy(interpolated_noise2_probs)

    # Reconstruct the signal and noise
    recon_signal_audio = waveform[0] * signal_probs_tensor
    recon_noise1_audio = waveform[0] * noise1_probs_tensor
    recon_noise2_audio = waveform[0] * noise2_probs_tensor




    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(recon_signal_audio)
    ax.plot(recon_noise1_audio)
    ax.plot(recon_noise2_audio)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Reconstructed Waveform')
    ax.grid(True)

    return fig

def plot_vad(audio_file_path):
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.0, "min_duration_off": 0.0}
    pipeline.instantiate(HYPER_PARAMETERS)
    waveform, sample_rate = torchaudio.load(audio_file_path)
    audio_duration = waveform.shape[1] / sample_rate
    SAMPLE_CHUNK = Segment(0, audio_duration)
    vad = pipeline(audio_file_path)

    fig, ax = plt.subplots(figsize=(12, 3))
    for segment in vad.get_timeline():
        ax.axvspan(segment.start, segment.end, color='red', alpha=0.5)
    ax.set_title('Voice Activity Detection')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Activity')
    ax.grid(True)

    return fig

def plot_signals(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    audio_duration = waveform.shape[1] / sample_rate
    SAMPLE_CHUNK = Segment(0, audio_duration)

    output = model(waveform.unsqueeze(1))
    _output = output.detach()[0].numpy()

    window_size = _output.shape[0]
    frame_duration = SAMPLE_CHUNK.duration / window_size
    frame_step = frame_duration

    shifted_frames = SlidingWindow(start=SAMPLE_CHUNK.start, duration=frame_duration, step=frame_step)
    output_sliding_window = SlidingWindowFeature(_output, shifted_frames)

    time_axis_output = np.linspace(start=SAMPLE_CHUNK.start, stop=SAMPLE_CHUNK.start + SAMPLE_CHUNK.duration, num=window_size, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 3))
    if _output.ndim > 1 and _output.shape[1] > 1:
        ax.plot(time_axis_output, _output[:, 0], label=f"Speech")
        ax.plot(time_axis_output, _output[:, 1], label=f"Noise 1")
        ax.plot(time_axis_output, _output[:, 2], label=f"Noise 2")
    else:
        ax.plot(time_axis_output, _output, label="Signal")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.set_title('Signal Presence Probability')
    ax.grid(True)
    ax.legend()

    return fig

def recon_plot_signals(audio_file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Calculate the time values for the x-axis
    time_values = np.arange(waveform.shape[1]) / sample_rate
    
    # Calculate the duration of the audio file in seconds
    audio_duration = waveform.shape[1] / sample_rate

    # Create a Segment object that spans the entire duration of the audio
    SAMPLE_CHUNK = Segment(0, audio_duration)

    # Voice Activity Detection pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_file_path)

    # Apply the model on the waveform
    output = model(waveform.unsqueeze(1))

    # Detach the output from the computation graph to avoid backpropagation
    _output = output.detach()[0].numpy()

    # Obtain the size of the output window
    window_size = _output.shape[0]

    # Calculate the duration of frames in seconds based on the output window size
    frame_duration = SAMPLE_CHUNK.duration / window_size

    # Create a time axis in seconds for the output frames
    time_axis_output = np.linspace(start=SAMPLE_CHUNK.start, stop=SAMPLE_CHUNK.start + SAMPLE_CHUNK.duration, num=window_size, endpoint=False)

    # Assuming _output has two columns: first for signal probabilities and second for noise probabilities
    signal_probs = _output[:, 0]
    noise1_probs = _output[:, 1]
    noise2_probs = _output[:, 2]

    # Interpolate probabilities using spline interpolation to match the length of the waveform
    interpolated_signal_probs = scipy.interpolate.interp1d(np.arange(len(signal_probs)), signal_probs, kind='cubic')(np.linspace(0, len(signal_probs) - 1, waveform.shape[1]))
    interpolated_noise1_probs = scipy.interpolate.interp1d(np.arange(len(noise1_probs)), noise1_probs, kind='cubic')(np.linspace(0, len(noise1_probs) - 1, waveform.shape[1]))
    interpolated_noise2_probs = scipy.interpolate.interp1d(np.arange(len(noise2_probs)), noise2_probs, kind='cubic')(np.linspace(0, len(noise2_probs) - 1, waveform.shape[1]))

    # Convert interpolated probabilities to PyTorch tensors
    signal_probs_tensor = torch.from_numpy(interpolated_signal_probs)
    noise1_probs_tensor = torch.from_numpy(interpolated_noise1_probs)
    noise2_probs_tensor = torch.from_numpy(interpolated_noise2_probs)

    # Reconstruct the signal and noise
    recon_signal_audio = waveform[0] * signal_probs_tensor
    recon_noise1_audio = waveform[0] * noise1_probs_tensor
    recon_noise2_audio = waveform[0] * noise2_probs_tensor

    time_axis_output = np.linspace(start=SAMPLE_CHUNK.start, stop=SAMPLE_CHUNK.start + SAMPLE_CHUNK.duration, num=window_size, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 3))
    if _output.ndim > 1 and _output.shape[1] > 1:
        ax.plot(interpolated_signal_probs, label=f"Speech")
        ax.plot(interpolated_noise1_probs, label=f"Noise 1")
        ax.plot(interpolated_noise2_probs, label=f"Noise 2")
    else:
        ax.plot(time_axis_output, _output, label="Signal")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.set_title('Interpolated Probabilities')
    ax.grid(True)
    ax.legend()

    return fig



#####################  Text



import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor

# Charger les modèles une seule fois
def load_models():
    model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sentiment_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    sentiment_analysis = pipeline('sentiment-analysis', model=sentiment_model_name)

    return tokenizer, model, sentiment_analysis

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def calculate_similarity(row, tokenizer, model):
    question_vec = encode_text(row['question'], tokenizer, model)
    response_vec = encode_text(row['answer'], tokenizer, model)
    similarity = cosine_similarity(question_vec, response_vec)[0][0]
    return similarity

def analyze_sentiment(row, sentiment_analysis):
    sentiment = sentiment_analysis(row['answer'])[0]
    return sentiment['label'], sentiment['score']

def process_row(row, tokenizer, model, sentiment_analysis):
    similarity = calculate_similarity(row, tokenizer, model)
    sentiment_label, sentiment_score = analyze_sentiment(row, sentiment_analysis)
    return similarity, sentiment_label, sentiment_score

def calculate_satisfaction_rate(df):
    total_responses = len(df)
    positive_responses = df[df['sentiment'] == 'POSITIVE'].shape[0]
    satisfaction_rate = (positive_responses / total_responses) * 100
    return satisfaction_rate

def calculate_correct_responses_percentage(df, threshold_similarity=0.4):
    total_responses = len(df)
    correct_responses = df[df['similarity'] > threshold_similarity].shape[0]
    correct_responses_percentage = (correct_responses / total_responses) * 100
    return correct_responses_percentage

# Add the new function process_text_data here
def process_text_data(data):
    # Load the models
    tokenizer, model, sentiment_analysis = load_models()

    # Process each row in the DataFrame
    results = []
    for _, row in data.iterrows():
        processed_row = process_row(row, tokenizer, model, sentiment_analysis)
        results.append(processed_row)

    # Add results to DataFrame
    data['similarity'], data['sentiment'], data['sentiment_score'] = zip(*results)

    # Optionally, you can also compute and add satisfaction and correctness metrics
    satisfaction_rate = calculate_satisfaction_rate(data)
    correctness_percentage = calculate_correct_responses_percentage(data)

    return data, satisfaction_rate, correctness_percentage


def main():
    # Charger le fichier CSV
    df = pd.read_csv('exemple (1).csv')

    # Charger les modèles
    tokenizer, model, sentiment_analysis = load_models()

    # Ajouter une barre de progression
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_row, row, tokenizer, model, sentiment_analysis)
            for row in df.to_dict(orient='records')
        ]
        for future in tqdm(futures, total=len(df)):
            results.append(future.result())

    # Ajouter les résultats au DataFrame
    df['similarity'], df['sentiment'], df['sentiment_score'] = zip(*results)

    # Calculer le taux de satisfaction
    satisfaction_rate = calculate_satisfaction_rate(df)
    print(f"Taux de satisfaction : {satisfaction_rate:.2f}%")

    # Calculer le pourcentage des réponses correctes
    correct_responses_percentage = calculate_correct_responses_percentage(df)
    print(f"Pourcentage des réponses correctes : {correct_responses_percentage:.2f}%")

    # Enregistrer les résultats dans un nouveau fichier CSV
    df.to_csv('interaction_results.csv', index=False)

    print("Analyse terminée. Les résultats ont été enregistrés dans 'interaction_results.csv'.")

if __name__ == "__main__":
    main()
