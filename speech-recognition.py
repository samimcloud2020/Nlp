import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('popular')  # Ensures all common models, including punkt_tab, are available

import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize speech recognizer and sentiment analyzer
recognizer = sr.Recognizer()
sia = SentimentIntensityAnalyzer()

# Speech recognition from microphone input
print("Please speak into the microphone...")

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.5)  # (Optional) ambient noise adjustment
    audio = recognizer.listen(source)

try:
    # Convert speech to text
    transcript = recognizer.recognize_google(audio)
    print("Transcribed Text:", transcript)

    # Tokenization (NLP step)
    tokens = word_tokenize(transcript)
    print("Tokens:", tokens)

    # Sentiment Analysis (NLP step)
    sentiment = sia.polarity_scores(transcript)
    print("Sentiment Scores:", sentiment)

except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Google API error: {e}")
