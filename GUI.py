import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from tkinter import *
from PIL import Image, ImageTk
from train_model import SimpleCNN
from datetime import datetime

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=len(emotion_dict))
model.load_state_dict(torch.load('assets/model_emocje.pth', map_location=device))
model.eval()

#Prediction of emotions
def predict_emotion(face_img):
    transform = transforms.Compose([                    #Creating chain of transofrm
        transforms.ToPILImage(),                        #Converting image from tensor to PIL format
        transforms.Grayscale(),                         #Converting color image to grayscale - only one canal(1)
        transforms.Resize((48, 48)),                    #Resizing to size of FER2013 scale
        transforms.ToTensor(),                          #Converting from PIL with values of pixels [0, 255] to tensor with values [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  #Normalizing from [0,1] to [1, -1]
    ])
    img = transform(face_img).unsqueeze(0).to(device)   #Add one more dimension for model at position 0
    with torch.no_grad():                               #Turn off gradient cuz it's prediction, not a traning
        output = model(img)
        pred = torch.argmax(output, 1).item()      #Return index of class that activates most
    return pred

class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rozpoznawanie emocji")
        self.window.configure(bg="#1e1e1e")

        self.video = cv2.VideoCapture(0)
        self.last_detected_emotion = "Neutral"

        self.main_frame = Frame(window, bg="#1e1e1e")
        self.main_frame.pack(fill=BOTH, expand=True)

        # Podgląd
        self.left_frame = Frame(self.main_frame, bg="#1e1e1e")
        self.left_frame.pack(side=LEFT, expand=True)

        self.camera_label = Label(self.left_frame, bg="#1e1e1e")
        self.camera_label.pack(pady=20)

        self.capture_button = Canvas(self.left_frame, width=80, height=80, bg="#1e1e1e", highlightthickness=0)
        self.capture_button.pack(pady=20)
        self.capture_button.create_oval(10, 10, 70, 70, outline="#ffffff", width=4)
        self.capture_button.bind("<Button-1>", self.capture_image)

        # Galeria
        self.right_frame = Frame(self.main_frame, bg="#2e2e2e")
        self.right_frame.pack(side=RIGHT, fill=Y, padx=10)

        Label(self.right_frame, text="Ostatnio wykryta emocja:", fg="white", bg="#2e2e2e", font=("Helvetica", 12)).pack(pady=5)
        self.emotion_label = Label(self.right_frame, text="", fg="cyan", bg="#2e2e2e", font=("Helvetica", 14, "bold"))
        self.emotion_label.pack()

        Label(self.right_frame, text="Galeria emocji:", fg="white", bg="#2e2e2e", font=("Helvetica", 12)).pack(pady=10)
        self.gallery_panel = Listbox(self.right_frame, width=30, height=20, bg="#1e1e1e", fg="white")
        self.gallery_panel.pack(pady=5, fill=BOTH, expand=True)
        self.gallery_panel.bind("<<ListboxSelect>>", self.preview_selected_image)

        self.preview_label = Label(self.right_frame, bg="#2e2e2e")
        self.preview_label.pack(pady=5)

        self.update_frame()
        self.update_gallery()

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                if face_img.size > 0:
                    emotion_idx = predict_emotion(face_img)
                    self.last_detected_emotion = emotion_dict[emotion_idx]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, self.last_detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            self.emotion_label.config(text=self.last_detected_emotion)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    def detect_faces(self, gray):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade.detectMultiScale(gray, 1.3, 5)

    def capture_image(self, event=None):
        ret, frame = self.video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                if face_img.size > 0:
                    emotion_idx = predict_emotion(face_img)
                    emotion = emotion_dict[emotion_idx]
                    dir_path = os.path.join("gallery", emotion)
                    os.makedirs(dir_path, exist_ok=True)
                    filename = f"{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    path = os.path.join(dir_path, filename)
                    cv2.imwrite(path, face_img)
                    self.last_detected_emotion = emotion
                    self.emotion_label.config(text=emotion)
                    self.update_gallery()
                    break

    def update_gallery(self):
        self.gallery_panel.delete(0, END)
        emotion = self.last_detected_emotion
        path = os.path.join("gallery", emotion)
        if os.path.exists(path):
            for file in sorted(os.listdir(path), reverse=True):
                if file.endswith(".png"):
                    self.gallery_panel.insert(END, file)

    def preview_selected_image(self, event=None):
        selection = self.gallery_panel.curselection()
        if selection:
            index = selection[0]
            filename = self.gallery_panel.get(index)
            emotion = self.last_detected_emotion
            filepath = os.path.join("gallery", emotion, filename)
            if os.path.exists(filepath):
                img = Image.open(filepath).resize((150, 150))
                imgtk = ImageTk.PhotoImage(img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk)

if __name__ == "__main__":
    root = Tk()
    root.configure(bg="#1e1e1e")
    app = EmotionApp(root)
    root.mainloop()
