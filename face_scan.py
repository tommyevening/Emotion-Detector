import cv2
import torch
import numpy as np
from torchvision import transforms
from tkinter import *
from PIL import Image, ImageTk

# Załaduj wytrenowany model
model = torch.load('model_emocje.pth', map_location=torch.device('cpu'))
model.eval()

# Funkcja do predykcji emocji
def predict_emotion(face_img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()
    return pred

# Mapowanie indeksów na emocje
emotion_dict = {0: "Złość", 1: "Wstręt", 2: "Strach", 3: "Szczęście", 4: "Smutek", 5: "Zaskoczenie", 6: "Neutralny"}

# GUI
class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rozpoznawanie emocji")
        self.video = cv2.VideoCapture(0)

        self.left_panel = Label(window)
        self.left_panel.pack(side=LEFT)

        self.right_panel = Text(window, width=30, height=20)
        self.right_panel.pack(side=RIGHT)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                emotion_idx = predict_emotion(face_img)
                emotion = emotion_dict[emotion_idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                self.right_panel.insert(END, f"Wykryto emocję: {emotion}\n")
                self.right_panel.see(END)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.left_panel.imgtk = imgtk
            self.left_panel.configure(image=imgtk)
        self.window.after(10, self.update_frame)

root = Tk()
app = EmotionApp(root)
root.mainloop()