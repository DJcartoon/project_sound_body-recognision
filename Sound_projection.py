import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import tensorflow as tf
import sys

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Audio settings
fs = 44100  # Sample rate
buffer_duration = 0.1  # Duration of the buffer in seconds
buffer_size = int(fs * buffer_duration)
buffer = np.zeros(buffer_size)

# TensorFlow model (e.g., custom gesture recognition model)
# Load your custom model here, such as modify sound based on hand gesture
# model = tf.keras.models.load_model('path_to_your_model')

# Function to generate a sine wave
def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

# Audio callback function
def audio_callback(outdata, frames, time, status):
    global buffer
    if status:
        print(status, file=sys.stderr)
    
    # Adjust buffer size to match frames
    buffer_size = len(outdata)
    if len(buffer) != buffer_size:
        buffer = np.zeros(buffer_size)
    
    outdata[:] = buffer.reshape(-1, 1)

# OpenCV window for hand tracking
cap = cv2.VideoCapture(0)

# Start audio stream with callback
stream = sd.OutputStream(samplerate=fs, channels=1, callback=audio_callback)
stream.start()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the x-coordinate of the index finger tip (landmark 8)
                x = hand_landmarks.landmark[8].x

                # Map the x-coordinate to a frequency range (e.g., 200 Hz to 2000 Hz)
                frequency = 200 + x * 1800

                # Generate a sine wave with the calculated frequency
                sine_wave = generate_sine_wave(frequency, buffer_duration, fs)

                # Update the buffer with the generated sine wave
                buffer[:len(sine_wave)] = sine_wave[:len(buffer)]

                # Optional: Prepare data for TensorFlow model
                # Example: Extract all landmarks and reshape for model input
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                landmarks = landmarks.flatten().reshape(1, -1)

                # Run TensorFlow model (if available)
                # predictions = model.predict(landmarks)
                # Use predictions to modify sound or other behaviors

                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    stream.stop()
    stream.close()
