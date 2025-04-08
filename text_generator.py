import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import requests
from tqdm import tqdm
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextGenerator:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        self.word_to_index = {}
        self.index_to_word = {}
        
    def download_shakespeare(self):
        """Download Shakespeare's text from Project Gutenberg."""
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        response = requests.get(url)
        return response.text
        
    def preprocess_text(self, text):
        """Preprocess the text by converting to lowercase and removing punctuation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
        
    def create_sequences(self, tokens):
        """Create sequences for training."""
        # Create word to index mapping
        unique_words = sorted(set(tokens))
        self.word_to_index = {word: index for index, word in enumerate(unique_words)}
        self.index_to_word = {index: word for index, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)
        
        # Create sequences
        sequences = []
        for i in range(len(tokens) - self.sequence_length):
            sequence = tokens[i:i + self.sequence_length]
            target = tokens[i + self.sequence_length]
            sequences.append((sequence, target))
            
        return sequences
        
    def build_model(self):
        """Build the LSTM model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 256, input_length=self.sequence_length),
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def prepare_sequences(self, sequences):
        """Prepare sequences for training."""
        X = []
        y = []
        
        for sequence, target in sequences:
            X.append([self.word_to_index[word] for word in sequence])
            y.append(self.word_to_index[target])
            
        return np.array(X), np.array(y)
        
    def train(self, epochs=50, batch_size=64):
        """Train the model."""
        # Download and preprocess the text
        text = self.download_shakespeare()
        tokens = self.preprocess_text(text)
        sequences = self.create_sequences(tokens)
        
        # Prepare the data
        X, y = self.prepare_sequences(sequences)
        
        # Build and train the model
        self.build_model()
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        return history
        
    def generate_text(self, seed_text, num_words=100):
        """Generate text based on a seed text."""
        if not self.model:
            raise ValueError("Model needs to be trained first!")
            
        # Preprocess the seed text
        seed_tokens = self.preprocess_text(seed_text)
        
        # Ensure we have enough tokens
        if len(seed_tokens) < self.sequence_length:
            raise ValueError(f"Seed text must contain at least {self.sequence_length} words")
            
        # Get the last sequence_length tokens
        current_sequence = seed_tokens[-self.sequence_length:]
        
        generated_text = seed_tokens.copy()
        
        for _ in range(num_words):
            # Convert current sequence to indices
            sequence_indices = [self.word_to_index[word] for word in current_sequence]
            sequence_indices = np.array([sequence_indices])
            
            # Predict the next word
            predicted_probs = self.model.predict(sequence_indices, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            predicted_word = self.index_to_word[predicted_index]
            
            # Add the predicted word to the generated text
            generated_text.append(predicted_word)
            
            # Update the current sequence
            current_sequence = current_sequence[1:] + [predicted_word]
            
        return ' '.join(generated_text)

def main():
    # Create and train the model
    generator = TextGenerator(sequence_length=50)
    print("Training the model...")
    history = generator.train(epochs=50)
    
    # Generate some text
    seed_text = "To be or not to be"
    print("\nGenerating text from seed:", seed_text)
    generated_text = generator.generate_text(seed_text, num_words=100)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main() 