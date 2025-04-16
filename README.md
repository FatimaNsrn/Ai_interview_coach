# Ai_interview_coach
AI Interview Coach

This project is a lightweight AI-powered interview coach built with Gradio and deployed on Hugging Face Spaces. It helps candidates practice answering interview questions by recording their responses, transcribing them using an ASR model (Whisper), and providing instant AI-generated feedback.

Features

Records spoken answers to interview questions

Transcribes audio to text using Whisper

Uses a language model to evaluate answers based on quality

Provides qualitative feedback

Completely browser-based (no installation needed)


Tech Stack

Python

Gradio (UI framework)

Hugging Face Transformers

Whisper (automatic speech recognition)

Hosted on Hugging Face Spaces (CPU environment)


Try the App

You can try the live demo here:
Live Demo

How to Run Locally

Clone the repo and install dependencies:

git clone https://huggingface.co/spaces/fatima98/practice_interview_Ai
pip install -r requirements.txt
python app.py

Project Goals

This project was created to demonstrate:

Practical use of ASR and LLMs in a real-time feedback loop

Building and deploying AI apps using Hugging Face and Gradio

Experience in prompt engineering and model selection under resource constraints
