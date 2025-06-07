# Emotion Detector App 

Real-time emotion detection application using computer vision and deep learning. The app captures video from your webcam, detects faces, and classifies emotions in real-time using a custom CNN model.

## Features

- **Real-time emotion detection** from webcam feed
- **7 emotion categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Live GUI interface** with video display and emotion log
- **Custom CNN model** trained for emotion classification
- **Face detection** using OpenCV Haar cascades

## Demo 

The application displays:
- Live video feed with face detection rectangles
- Real-time emotion labels on detected faces
- Text log of detected emotions in the side panel

## Tech Stack 

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision and face detection
- **Tkinter** - GUI framework
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical computations

## Model Architecture 

Custom CNN with:
- 2 Convolutional layers (32, 64 filters)
- MaxPooling layers
- Fully connected layers (128 neurons)
- Dropout for regularization
- Input: 48x48 grayscale images
- Output: 7 emotion classes

## Installation 

1. **Clone the repository**
```bash
git clone https://github.com/tommyevening/Emotion-Detector-app.git
cd Emotion-Detector
```

2. **Install dependencies**
```bash
pip install torch torchvision opencv-python pillow numpy
```

3. **Prepare the model**
   - Train your model using `train_model.py` or
   - Place pre-trained model weights in `assets/model_emocje.pth`

## Usage 

### Training the Model
```bash
python train_model.py
```
Make sure your dataset is organized as:
```
assets/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### Running the Application
```bash
python main.py
```

## Dataset 

The model is designed to work with FER-2013 format:
- **Image size**: 48x48 pixels
- **Color**: Grayscale
- **Classes**: 7 emotions
- **Format**: Standard image files (jpg, png)

## Model Performance 📈

- **Input resolution**: 48x48 grayscale
- **Training epochs**: 10 (configurable)
- **Batch size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss function**: CrossEntropyLoss

## File Structure 

```
emotion-recognition-app/
├── main.py              # Main application with GUI
├── train_model.py       # Model training script
├── assets/
│   ├── model_emocje.pth # Trained model weights
│   ├── train/           # Training dataset
│   └── test/            # Test dataset
└── README.md
```

## Contributing 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements 

- [ ] Add more emotion categories
- [ ] Implement data augmentation
- [ ] Add confidence scores display
- [ ] Support for multiple face tracking
- [ ] Model optimization for better accuracy
- [ ] Export emotion statistics
- [ ] Add dark/light theme toggle

## Acknowledgments 

- OpenCV for computer vision capabilities
- PyTorch team for the deep learning framework
- FER-2013 dataset contributors
- Haar cascade classifiers for face detection

---
