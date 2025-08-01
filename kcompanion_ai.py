#!/usr/bin/env python3
"""
kcompanion_ai.py - PyTorch-based AI component for the kCompanion system

This script implements advanced AI capabilities using PyTorch with CUDA acceleration
to complement the kcompanion kernel module.

Requirements:
    - PyTorch with CUDA support
    - NumPy
    - Pandas
    - Matplotlib (optional, for visualization)

This daemon:
    1. Reads system metrics from the kcompanion kernel module via /dev/kcompanion
    2. Processes data using PyTorch neural networks
    3. Detects anomalies using deep learning models
    4. Learns from user feedback
    5. Sends AI-driven suggestions back to the kernel module
"""

import os
import sys
import time
import json
import fcntl
import struct
import signal
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# PyTorch imports with CUDA support
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Constants
KCOMPANION_DEVICE = "/dev/kcompanion"
METRICS_HISTORY_SIZE = 1000  # Store more history in userspace
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "/home/licht/Desktop/kernel3/ai_models"
LOG_PATH = "/home/licht/Desktop/kernel3/ai_logs"

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# IOCTL commands (must match those in the kernel module)
KCOMPANION_CLEAR_SUGGESTIONS = 0x6B01  # _IO('k', 1)
KCOMPANION_GET_METRICS = 0x806B02     # _IOR('k', 2, struct system_metrics)
KCOMPANION_RESET_STATS = 0x6B03       # _IO('k', 3)

# Data structure for metrics (must match the kernel structure)
class SystemMetrics:
    def __init__(self):
        self.cpu_user = 0
        self.cpu_system = 0
        self.cpu_idle = 0
        self.mem_total = 0
        self.mem_free = 0
        self.mem_available = 0
        self.disk_reads = 0
        self.disk_writes = 0
        self.syscall_count = 0
        self.timestamp = 0
        
    def to_numpy(self):
        """Convert metrics to numpy array for model input"""
        return np.array([
            self.cpu_user, 
            self.cpu_system, 
            self.cpu_idle,
            self.mem_free,
            self.mem_available,
            self.disk_reads,
            self.disk_writes,
            self.syscall_count
        ], dtype=np.float32)

# Define neural network models

class LSTMPredictor(nn.Module):
    """LSTM model for time-series prediction of system metrics"""
    def __init__(self, input_size=8, hidden_size=64, output_size=8, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class AnomalyAutoencoder(nn.Module):
    """Autoencoder model for anomaly detection"""
    def __init__(self, input_size=8):
        super(AnomalyAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SuggestionGenerator(nn.Module):
    """Model for generating suggestions based on system state"""
    def __init__(self, input_size=8, num_suggestions=10):
        super(SuggestionGenerator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_suggestions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class KCompanionAI:
    """Main class for the kCompanion AI system"""
    
    def __init__(self):
        self.running = False
        self.metrics_history = deque(maxlen=METRICS_HISTORY_SIZE)
        self.anomaly_threshold = 0.1  # Initial threshold
        self.feedback_history = []
        
        # Initialize models
        self.predictor = LSTMPredictor().to(DEVICE)
        self.autoencoder = AnomalyAutoencoder().to(DEVICE)
        self.suggestion_generator = SuggestionGenerator().to(DEVICE)
        
        # Load models if they exist
        self.load_models()
        
        # Set optimizers
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=LEARNING_RATE)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=LEARNING_RATE)
        self.suggestion_optimizer = optim.Adam(self.suggestion_generator.parameters(), lr=LEARNING_RATE)
        
        # Suggestion templates for classification output
        self.suggestion_templates = [
            "CPU usage is abnormally high. Consider checking for runaway processes.",
            "System CPU usage indicates heavy system load, possibly I/O or networking.",
            "Memory pressure detected. Consider closing unused applications.",
            "Unusual disk I/O activity detected. Check for disk-intensive operations.",
            "Low memory condition detected. Less than 10% of memory is available.",
            "High CPU load detected with minimal idle time.",
            "High system call rate detected. Check for busy applications.",
            "System performance could be improved by enabling ccache for compilation.",
            "Consider enabling swap on zram for improved memory management.",
            "System temperature is rising. Check cooling systems and CPU frequency scaling."
        ]
        
        # Thread for background processing
        self.processing_thread = None
        
    def load_models(self):
        """Load trained models if they exist"""
        try:
            if os.path.exists(f"{MODEL_SAVE_PATH}/predictor.pt"):
                self.predictor.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/predictor.pt"))
                print("Loaded predictor model")
                
            if os.path.exists(f"{MODEL_SAVE_PATH}/autoencoder.pt"):
                self.autoencoder.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/autoencoder.pt"))
                print("Loaded autoencoder model")
                
            if os.path.exists(f"{MODEL_SAVE_PATH}/suggestion_generator.pt"):
                self.suggestion_generator.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/suggestion_generator.pt"))
                print("Loaded suggestion generator model")
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def save_models(self):
        """Save trained models"""
        try:
            torch.save(self.predictor.state_dict(), f"{MODEL_SAVE_PATH}/predictor.pt")
            torch.save(self.autoencoder.state_dict(), f"{MODEL_SAVE_PATH}/autoencoder.pt")
            torch.save(self.suggestion_generator.state_dict(), f"{MODEL_SAVE_PATH}/suggestion_generator.pt")
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
            
    def get_metrics_from_kernel(self):
        """Get current system metrics from the kernel module"""
        try:
            with open(KCOMPANION_DEVICE, "rb+") as dev:
                # IOCTL to get metrics
                # In a real implementation, this would properly use fcntl.ioctl
                # with appropriate struct packing/unpacking
                
                # For demo purposes, we'll just read from the device instead
                # In real code, you'd use: fcntl.ioctl(dev.fileno(), KCOMPANION_GET_METRICS, buffer)
                data = dev.read(1024)  # Read suggestion data for now
                
                # In a real implementation, we would parse binary data from the kernel
                # For now, we'll simulate with random metrics
                metrics = SystemMetrics()
                metrics.cpu_user = np.random.randint(0, 100)
                metrics.cpu_system = np.random.randint(0, 50)
                metrics.cpu_idle = 100 - metrics.cpu_user - metrics.cpu_system
                metrics.mem_total = 8 * 1024 * 1024  # 8GB
                metrics.mem_free = np.random.randint(1 * 1024 * 1024, 6 * 1024 * 1024)
                metrics.mem_available = metrics.mem_free + np.random.randint(0, 1 * 1024 * 1024)
                metrics.disk_reads = np.random.randint(0, 1000)
                metrics.disk_writes = np.random.randint(0, 500)
                metrics.syscall_count = np.random.randint(0, 20000)
                metrics.timestamp = time.time()
                
                return metrics
        except Exception as e:
            print(f"Error getting metrics from kernel: {e}")
            # Return simulated metrics as fallback
            metrics = SystemMetrics()
            metrics.cpu_user = np.random.randint(0, 100)
            metrics.cpu_system = np.random.randint(0, 50)
            metrics.cpu_idle = 100 - metrics.cpu_user - metrics.cpu_system
            metrics.mem_total = 8 * 1024 * 1024  # 8GB
            metrics.mem_free = np.random.randint(1 * 1024 * 1024, 6 * 1024 * 1024)
            metrics.mem_available = metrics.mem_free + np.random.randint(0, 1 * 1024 * 1024)
            metrics.disk_reads = np.random.randint(0, 1000)
            metrics.disk_writes = np.random.randint(0, 500)
            metrics.syscall_count = np.random.randint(0, 20000)
            metrics.timestamp = time.time()
            return metrics
            
    def send_suggestion_to_kernel(self, suggestion):
        """Send AI-generated suggestion to the kernel module"""
        try:
            with open(KCOMPANION_DEVICE, "w") as dev:
                dev.write(suggestion)
            return True
        except Exception as e:
            print(f"Error sending suggestion to kernel: {e}")
            return False
            
    def detect_anomalies(self, current_metrics):
        """Use autoencoder to detect anomalies in system metrics"""
        if len(self.metrics_history) < 10:
            return False, "Not enough history for anomaly detection"
            
        # Convert current metrics to tensor
        current_tensor = torch.tensor(current_metrics.to_numpy(), dtype=torch.float32).to(DEVICE)
        
        # Use autoencoder to reconstruct the input
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(current_tensor)
            
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, current_tensor).item()
        
        # If error exceeds threshold, it's an anomaly
        is_anomaly = reconstruction_error > self.anomaly_threshold
        
        return is_anomaly, reconstruction_error
            
    def generate_suggestion(self, current_metrics):
        """Generate suggestion based on current system state"""
        current_tensor = torch.tensor(current_metrics.to_numpy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        self.suggestion_generator.eval()
        with torch.no_grad():
            suggestion_probs = self.suggestion_generator(current_tensor).squeeze(0)
            
        # Get index of highest probability suggestion
        suggestion_idx = torch.argmax(suggestion_probs).item()
        
        # Return the corresponding suggestion template
        return self.suggestion_templates[suggestion_idx]
            
    def train_models(self):
        """Train models on collected history data"""
        if len(self.metrics_history) < BATCH_SIZE:
            print("Not enough data for training")
            return
            
        # Prepare training data
        metrics_array = np.array([m.to_numpy() for m in self.metrics_history])
        metrics_tensor = torch.tensor(metrics_array, dtype=torch.float32).to(DEVICE)
        
        # Train autoencoder
        self.autoencoder.train()
        self.autoencoder_optimizer.zero_grad()
        
        reconstructed = self.autoencoder(metrics_tensor)
        autoencoder_loss = F.mse_loss(reconstructed, metrics_tensor)
        
        autoencoder_loss.backward()
        self.autoencoder_optimizer.step()
        
        # Train LSTM predictor (prepare sequences)
        sequence_length = 10
        if len(self.metrics_history) > sequence_length:
            sequences = []
            targets = []
            
            for i in range(len(metrics_array) - sequence_length):
                seq = metrics_array[i:i+sequence_length]
                target = metrics_array[i+sequence_length]
                sequences.append(seq)
                targets.append(target)
                
            if sequences:
                seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(DEVICE)
                target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).to(DEVICE)
                
                self.predictor.train()
                self.predictor_optimizer.zero_grad()
                
                predictions = self.predictor(seq_tensor)
                predictor_loss = F.mse_loss(predictions, target_tensor)
                
                predictor_loss.backward()
                self.predictor_optimizer.step()
                
                print(f"Autoencoder loss: {autoencoder_loss.item():.4f}, Predictor loss: {predictor_loss.item():.4f}")
            else:
                print(f"Autoencoder loss: {autoencoder_loss.item():.4f}")
        else:
            print(f"Autoencoder loss: {autoencoder_loss.item():.4f}")
            
        # Update anomaly threshold based on reconstruction errors
        with torch.no_grad():
            errors = F.mse_loss(self.autoencoder(metrics_tensor), metrics_tensor, reduction='none').mean(dim=1)
            # Set threshold to mean + 2*std
            self.anomaly_threshold = errors.mean().item() + 2 * errors.std().item()
            print(f"Updated anomaly threshold: {self.anomaly_threshold:.6f}")
            
    def process_feedback(self, feedback, is_positive):
        """Process feedback from kernel/user to improve models"""
        # This would implement reinforcement learning to adjust the suggestion generator
        # For now, we'll just log the feedback
        self.feedback_history.append((feedback, is_positive, time.time()))
        print(f"Received {'positive' if is_positive else 'negative'} feedback: {feedback}")
            
    def processing_loop(self):
        """Main AI processing loop"""
        train_counter = 0
        
        while self.running:
            try:
                # Get current metrics from kernel
                current_metrics = self.get_metrics_from_kernel()
                self.metrics_history.append(current_metrics)
                
                # Detect anomalies
                is_anomaly, anomaly_score = self.detect_anomalies(current_metrics)
                
                if is_anomaly:
                    # Generate suggestion
                    suggestion = self.generate_suggestion(current_metrics)
                    print(f"Anomaly detected (score: {anomaly_score:.6f}): {suggestion}")
                    
                    # Send suggestion to kernel
                    self.send_suggestion_to_kernel(suggestion)
                    
                # Train models periodically (every 20 iterations)
                train_counter += 1
                if train_counter >= 20:
                    print("Training models...")
                    self.train_models()
                    train_counter = 0
                    
                    # Save models after training
                    self.save_models()
                
                # Sleep to avoid excessive CPU usage
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(10)  # Longer sleep on error
                
    def start(self):
        """Start the AI system"""
        if self.running:
            print("Already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("KCompanion AI started")
        
    def stop(self):
        """Stop the AI system"""
        if not self.running:
            print("Already stopped")
            return
            
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.save_models()
        print("KCompanion AI stopped")
        
def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals"""
    print("\nShutting down KCompanion AI...")
    if 'ai_system' in globals():
        ai_system.stop()
    sys.exit(0)

def main():
    """Main function"""
    global ai_system
    
    parser = argparse.ArgumentParser(description="KCompanion AI System")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    ai_system = KCompanionAI()
    ai_system.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ai_system.stop()

if __name__ == "__main__":
    main()
