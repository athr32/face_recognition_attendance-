import tkinter as tk
from tkinter import ttk, Message
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# Initialize Window
window = tk.Tk()
window.title("Face Recognizer Attendance System")
window.geometry('1366x768')
window.configure(bg="#1e1e2d")  # Dark theme background

# Ensure necessary directories exist
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("ImagesUnknown", exist_ok=True)

# UI Styling
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 14, "bold"), background="#1e1e2d", foreground="white")
style.configure("TEntry", font=("Helvetica", 14), padding=5)
style.configure("TButton", font=("Helvetica", 12, "bold"), padding=6, relief="raised", background="#28a745")

# UI Components
lbl = ttk.Label(window, text="EMPLOYEE ID")
lbl.place(x=400, y=200)

txt = ttk.Entry(window, width=25)
txt.place(x=700, y=205)

lbl2 = ttk.Label(window, text="EMPLOYEE NAME")
lbl2.place(x=400, y=300)

txt2 = ttk.Entry(window, width=25)
txt2.place(x=700, y=305)

message = ttk.Label(window, text="", width=40, anchor="center", foreground="#ffcc00")
message.place(x=700, y=400)

# Functions
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def TakeImages():
    Id = txt.get()
    name = txt2.get()

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        last_image = None
        last_gray = None

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                last_image = img.copy()
                last_gray = gray[y:y+h, x:x+w]

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('Capturing Face', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        if last_image is not None and last_gray is not None:
            filename = os.path.join("TrainingImage", f"{name}.{Id}.jpg")
            cv2.imwrite(filename, last_gray)

            # **Check if ID already exists in EmployeeDetails.csv**
            try:
                df = pd.read_csv("EmployeeDetails.csv", header=None, names=['Id', 'Name'])
                if str(Id) not in df['Id'].astype(str).values:  # Convert ID column to string for comparison
                    with open('EmployeeDetails.csv', 'a+', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow([Id, name])
            except FileNotFoundError:
                with open('EmployeeDetails.csv', 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([Id, name])

            message.configure(text=f"Image & Details Saved for ID: {Id}, Name: {name}")
        else:
            message.configure(text="No Face Detected!")

    else:
        message.configure(text="Enter a valid Numeric ID and Alphabetical Name")


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Ids))
    recognizer.save(os.path.join("TrainingImageLabel", "Trainer.yml"))
    message.configure(text="Image Training Complete")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, Ids = [], []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join("TrainingImageLabel", "Trainer.yml"))
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        df = pd.read_csv("EmployeeDetails.csv", header=None, names=['Id', 'Name'])
        df['Id'] = df['Id'].astype(str)  
        df['Name'] = df['Name'].astype(str).str.strip()
    except Exception as e:
        print(f"Error reading EmployeeDetails.csv: {e}")
        return

    cam = cv2.VideoCapture(0)
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Get today's date

    # **Check if Attendance.csv exists and is not empty**
    if os.path.exists("Attendance.csv") and os.path.getsize("Attendance.csv") > 0:
        attendance = pd.read_csv("Attendance.csv")
    else:
        attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])

    already_marked = set(attendance[attendance['Date'] == today_date]['Id'].astype(str).values) if not attendance.empty else set()

    while True:
        ret, im = cam.read()
        if not ret:
            print("Error: Camera not working properly")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                Id = str(Id)
                if Id in df['Id'].astype(str).values:
                    name = df.loc[df['Id'].astype(str) == Id, 'Name'].values[0]
                else:
                    name = "Unknown"

                # **Check if this employee already has attendance today**
                if Id not in already_marked:
                    new_record = pd.DataFrame([[Id, name, date, timeStamp]], columns=['Id', 'Name', 'Date', 'Time'])
                    attendance = pd.concat([attendance, new_record], ignore_index=True)
                    already_marked.add(Id)
                    
                else:
                    message.configure(text=f"Attendance recorded for {name}")
            else:
                name = "Unknown"

            cv2.putText(im, f"{Id}-{name}", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    
    # **Save attendance only if new records were added**
    if not attendance.empty:
        attendance.to_csv("Attendance.csv", index=False)


# Hover Effects for Buttons
def on_enter(e):
    e.widget["background"] = "#218838"

def on_leave(e):
    e.widget["background"] = "#28a745"

# Button Creation Function
def create_button(text, command, x, y):
    btn = tk.Button(window, text=text, command=command, width=20, height=2, bg="#28a745", fg="white", font=("Arial", 12, "bold"))
    btn.place(x=x, y=y)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

# Buttons
create_button("Take Images", TakeImages, 200, 500)
create_button("Train Images", TrainImages, 500, 500)
create_button("Track Images", TrackImages, 800, 500)
create_button("Quit", window.quit, 1100, 500)

# Run Application
window.mainloop()
