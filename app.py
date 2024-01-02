import gradio as gr
import whisper
from transformers import pipeline

model = whisper.load_model("base")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def get_sentiment_emoji(sentiment):
    # Define the emojis corresponding to each sentiment
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text

def inference(audio, sentiment_option):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

    return lang.upper(), result.text, sentiment_output

title = """<h1 align="center">🎤 Multilingual ASR 💬</h1>"""
image_path = "thmbnail.jpg"
description = """
💻 This demo showcases a general-purpose speech recognition model called Whisper. It is trained on a large dataset of diverse audio and supports multilingual speech recognition, speech translation, and language identification tasks.<br><br>
<br>
⚙️ Components of the tool:<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Real-time multilingual speech recognition<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Language identification<br>
&nbsp;&nbsp;&nbsp;&nbsp; - Sentiment analysis of the transcriptions<br>
<br>
🎯 The sentiment analysis results are provided as a dictionary with different emotions and their corresponding scores.<br>
<br>

😃 The sentiment analysis results are displayed with emojis representing the corresponding sentiment.<br>
<br>

✅ The higher the score for a specific emotion, the stronger the presence of that emotion in the transcribed text.<br>
<br>

❓ Use the microphone for real-time speech recognition.<br>
<br>

⚡️ The model will transcribe the audio and perform sentiment analysis on the transcribed text.<br>

"""

custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

block = gr.Blocks(css=custom_css)

with block:
    gr.HTML(title)

    with gr.Row():
        with gr.Column():
            gr.Image(image_path, elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.HTML(description)

    with gr.Group():
        with gr.Box():
            audio = gr.Audio(
                label="Input Audio",
                show_label=False,
                source="microphone",
                type="filepath"
            )

            sentiment_option = gr.Radio(
                choices=["Sentiment Only", "Sentiment + Score"],
                label="Select an option",
                default="Sentiment Only"
            )

            btn = gr.Button("Transcribe")

        lang_str = gr.Textbox(label="Language")

        text = gr.Textbox(label="Transcription")

        sentiment_output = gr.Textbox(label="Sentiment Analysis Results", output=True)

        btn.click(inference, inputs=[audio, sentiment_option], outputs=[lang_str, text, sentiment_output])

        gr.HTML('''
        <div class="footer">
            <p>Model by <a href="https://github.com/openai/whisper" style="text-decoration: underline;" target="_blank">OpenAI</a>
            </p>
        </div>
        ''') 

block.launch()
