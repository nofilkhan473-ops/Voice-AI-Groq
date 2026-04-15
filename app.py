import os
import whisper
import gradio as gr
from groq import Groq
from gtts import gTTS

# --- SETUP CREDENTIALS ---
# Hugging Face automatically loads Secrets into os.environ
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Please add it to the Space Secrets.")

groq_client = Groq(api_key=api_key)

# --- LOAD WHISPER MODEL ---
print("Loading Whisper Model (small)...")
whisper_model = whisper.load_model("small")
print("Whisper Model loaded successfully!")

# --- MODULAR PIPELINE FUNCTIONS ---
def speech_to_text(audio_filepath):
    if audio_filepath is None:
        return ""
    result = whisper_model.transcribe(audio_filepath)
    return result["text"].strip()

def generate_llm_response(user_text):
    if not user_text:
        return "I didn't hear anything. Please try speaking again."
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly, helpful, and highly conversational AI assistant. "
                        "Keep your responses short, natural, and easy to speak aloud. "
                        "Do not use markdown formatting, code blocks, or long lists. "
                        "Talk like a human on a phone call."
                    )
                },
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=150,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error connecting to Groq: {str(e)}"

def text_to_speech(ai_text):
    output_filename = "ai_response.mp3"
    tts = gTTS(text=ai_text, lang='en', slow=False)
    tts.save(output_filename)
    return output_filename

def process_voice_input(audio_filepath):
    user_transcription = speech_to_text(audio_filepath)
    ai_text_response = generate_llm_response(user_transcription)
    output_audio_path = text_to_speech(ai_text_response)
    return user_transcription, ai_text_response, output_audio_path

# --- GRADIO FRONTEND INTERFACE ---
with gr.Blocks(title="Voice-to-Voice AI with LLaMA & Groq") as demo:
    gr.Markdown("# 🎙️ Ultra-Fast Voice AI Assistant")
    gr.Markdown("Speak into the microphone. Powered by **Whisper**, **Llama 3.3 (Groq)**, and **gTTS**.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Voice")
            submit_btn = gr.Button("Send to AI", variant="primary")
        
        with gr.Column():
            user_text_display = gr.Textbox(label="You said:")
            ai_text_display = gr.Textbox(label="AI Response:")
            audio_output = gr.Audio(label="AI Voice", autoplay=True)
            
    submit_btn.click(
        fn=process_voice_input,
        inputs=audio_input,
        outputs=[user_text_display, ai_text_display, audio_output]
    )

# Hugging Face Spaces requires this simple launch command
demo.launch()
