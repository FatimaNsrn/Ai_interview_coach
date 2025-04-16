import os
import gradio as gr
from transformers import pipeline

# Install necessary libraries
os.system('pip install --upgrade --no-cache-dir transformers==4.11.3')
os.system('pip install --upgrade --no-cache-dir torch==2.0.0')
os.system('pip install --upgrade --no-cache-dir openai-whisper==20230918')
os.system('pip install --upgrade --no-cache-dir gradio==3.0')

# Now import the libraries that depend on these installations
import whisper
import torch

# Load Whisper for speech-to-text
asr_model = whisper.load_model("base")

# Feedback generation model: DistilGPT-2
feedback_model = pipeline(
    "text-generation", 
    model="distilgpt2", 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1
)

# Interview questions
# Interview questions
questions = [
    "Tell me about yourself.",
    "Why do you want this internship?",
    "What are your strengths and weaknesses?",
    "What motivates you to perform well in your job?",
    "Where do you see yourself in 5 years?"
]
current_question = {"index": 0}

def get_question():
    """Returns the current interview question."""
    q = questions[current_question["index"]]
    current_question["index"] = (current_question["index"] + 1) % len(questions)
    return q

def process_audio(audio_file):
    """Process the audio file to get feedback."""
    if audio_file is None:
        return "Please record an answer.", ""

    # Convert audio to text
    result = asr_model.transcribe(audio_file)
    transcript = result["text"]

    # Generate feedback
    prompt = (
        "You are a professional interview evaluator. Evaluate the following candidate's answer strictly based on:\n"
        "1. Clarity\n2. Relevance\n3. Professionalism\n4. Structure\n\n"
        f"Candidate's answer: \"{transcript}\"\n\n"
        "Be honest and detailed. End your response with a score out of 100."
    )

    response = feedback_model(prompt, max_length=200)[0]["generated_text"]

    return response, transcript

# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## AI Interview Coach (Free, Hugging Face Version)")
    
    # Textbox for the interview question
    question_text = gr.Textbox(label="Interview Question", interactive=False)
    
    # Button to get the interview question
    get_q_btn = gr.Button("Get Interview Question")
    
    with gr.Row():
        # Audio input for the candidate's response
        audio_input = gr.Audio(source="microphone", type="filepath", label="Record Your Answer")
        submit_btn = gr.Button("Submit")
    
    # Textboxes for feedback and transcript
    feedback = gr.Textbox(label="AI Feedback")
    transcript = gr.Textbox(label="Transcript of Your Answer")

    # Set up button actions
    get_q_btn.click(get_question, outputs=question_text)
    submit_btn.click(process_audio, inputs=audio_input, outputs=[feedback, transcript])

# Run the app
if __name__ == "__main__":
    demo.launch()