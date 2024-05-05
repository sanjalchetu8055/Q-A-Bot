#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
import torch
from model import QAModel, text_to_sequence  # Ensure this import line is correct based on your model's location
import pandas as pd

app = Flask(__name__)

# Load your data
df = pd.read_excel('extended_final_augmented_qa_dataset.xlsx')
answers = pd.factorize(df['Answer'])[0]  # Convert answers to integer codes

# Load Model
model = QAModel(vocab_size=256, embedding_dim=50, hidden_dim=150, output_dim=len(pd.unique(answers)), dropout_rate=0.339, num_layers=2)
model.load_state_dict(torch.load('best_model.pth'))  # Ensure your model is saved and loaded correctly
model.eval()

# Routes
@app.route('/')
def home():
    questions = df['Question'].tolist()[:10]  # Select only the first 10 questions to display
    return render_template('index.html', questions=questions)

@app.route('/predict', methods=['POST'])
def predict():
    question = request.form.get('question')
    sequence = text_to_sequence([question])
    prediction = model(sequence)
    _, predicted_idx = torch.max(prediction, dim=1)
    answer = df['Answer'].iloc[predicted_idx.item()]  # Fetching the corresponding answer
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

