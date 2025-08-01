# kCompanion: AI-Driven System Companion
Basic installation sudo chmod +x setup.sh
sudo ./setup.sh
This project implements an AI-driven system companion for Linux with a hybrid architecture:

1. **Kernel Module**: Collects system metrics efficiently and provides real-time monitoring
2. **PyTorch AI Daemon**: Uses deep learning with CUDA for advanced anomaly detection

## Components

### Kernel Module (`kcompanion.c`)

- Collects system metrics (CPU, memory, I/O, syscalls)
- Provides a character device interface (`/dev/kcompanion`)
- Performs basic statistical analysis
- Sends data to userspace for advanced AI processing

### AI Daemon (`kcompanion_ai.py`)

- PyTorch-based deep learning with CUDA acceleration
- LSTM model for time-series prediction
- Autoencoder for anomaly detection
- Reinforcement learning for adapting to user feedback

## Requirements

- Linux kernel 6.8+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- Python 3.8+

## Installation

### Kernel Module

```bash
# Compile and install the kernel module
make
sudo insmod kcompanion.ko

# Verify it's loaded
lsmod | grep kcompanion
ls -l /dev/kcompanion
```

### PyTorch AI Daemon

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the AI daemon
sudo python3 kcompanion_ai.py
```

## Usage

The system works automatically once both components are running. The kernel module collects data and the AI daemon analyzes it.

To read current suggestions:
```bash
cat /dev/kcompanion
```

To provide feedback (helps the AI learn):
```bash
# Positive feedback
echo "good:This suggestion was helpful" > /dev/kcompanion

# Negative feedback
echo "bad:This suggestion was not useful" > /dev/kcompanion
```

## Architecture

```
┌────────────────────┐     ┌─────────────────────────┐
│   Kernel Module    │     │   PyTorch AI Daemon     │
│   (kcompanion.c)   │     │   (kcompanion_ai.py)    │
│                    │     │                         │
│  ┌──────────────┐  │     │  ┌─────────────────┐    │
│  │System Metrics│  │     │  │LSTM Predictor   │    │
│  │Collection    │  │     │  │(Time Series)    │    │
│  └──────┬───────┘  │     │  └────────┬────────┘    │
│         │          │     │           │             │
│  ┌──────┴───────┐  │     │  ┌────────┴────────┐    │
│  │Basic         │  │     │  │Anomaly          │    │
│  │Statistics    │◄─┼─────┼──┤Autoencoder      │    │
│  └──────┬───────┘  │     │  └────────┬────────┘    │
│         │          │     │           │             │
│  ┌──────┴───────┐  │     │  ┌────────┴────────┐    │
│  │Character     │◄─┼────►│  │Suggestion       │    │
│  │Device        │  │     │  │Generator        │    │
│  └──────────────┘  │     │  └─────────────────┘    │
└────────────────────┘     └─────────────────────────┘
```

## Performance Notes

- The AI daemon requires significant GPU resources for CUDA-accelerated deep learning
- The kernel module has minimal performance impact (< 1% CPU overhead)

## Security Considerations

- The kernel module runs with kernel privileges - use caution
- The character device has appropriate permission restrictions

## License

GPL v2 (required for kernel modules)

## How It Works

The AI-driven anomaly detection system works through:

1. **Data Collection**: The module periodically collects system metrics using kernel APIs
2. **Deep Learning Analysis**: 
   - The userspace AI component uses LSTM networks for time-series prediction
   - Autoencoders detect anomalies through reconstruction error
   - Neural networks classify system states to generate appropriate suggestions
3. **Adaptive Learning**:
   - The AI system continuously learns from historical data
   - User feedback is incorporated to improve future suggestions

## Uninstallation

1. Unload the module and stop the AI daemon:

```bash
sudo ./setup.sh stop
```

2. Completely uninstall:

```bash
sudo ./setup.sh uninstall
```

## Security Considerations

- The module is designed to be memory-safe and avoid kernel crashes
- It uses appropriate locking mechanisms (spinlocks, mutexes) to ensure thread safety
- Permission checks are in place to prevent unauthorized access

## Limitations

- GPU performance depends on your NVIDIA hardware capabilities
- Some metrics are currently simulated (placeholders) rather than actual system values
- Memory usage is fixed to avoid allocation issues in kernel space

## License

This software is licensed under the GPL license, as required for Linux kernel modules.
