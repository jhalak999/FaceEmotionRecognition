import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from deepface import DeepFace
import numpy as np
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

class EmotionRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Facial Emotion Recognition")
        self.window.geometry("1200x700")
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 12))
        
        # Variables
        self.is_capturing = False
        self.camera_index = 0
        self.current_emotion = "None"
        self.emotion_history = deque(maxlen=100)
        self.emotion_counts = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, 
                              "sad": 0, "surprise": 0, "neutral": 0}
        self.start_time = None
        self.record_data = False
        self.data_folder = "emotion_data"
        
        # Create data folder if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Main frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for video
        self.left_frame = ttk.Frame(main_frame, width=640, height=480)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Video canvas
        self.canvas = tk.Canvas(self.left_frame, width=640, height=480, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for controls and stats
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(right_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.IntVar(value=0)
        ttk.Radiobutton(camera_frame, text="0", variable=self.camera_var, value=0).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(camera_frame, text="1", variable=self.camera_var, value=1).pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(buttons_frame, text="Start", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.record_button = ttk.Button(buttons_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Current emotion display
        emotion_frame = ttk.LabelFrame(right_frame, text="Current Emotion")
        emotion_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.emotion_label = ttk.Label(emotion_frame, text="None", font=("Arial", 24))
        self.emotion_label.pack(padx=5, pady=10)
        
        self.confidence_label = ttk.Label(emotion_frame, text="Confidence: 0%")
        self.confidence_label.pack(padx=5, pady=5)
        
        # Session stats frame
        stats_frame = ttk.LabelFrame(right_frame, text="Session Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chart for emotion distribution
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=stats_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Video capture variable
        self.cap = None
        
        # Update the graph initially
        self.update_emotion_graph()
    
    def start_capture(self):
        if self.is_capturing:
            return
            
        # Get selected camera
        self.camera_index = self.camera_var.get()
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            self.status_var.set(f"Error: Could not open camera {self.camera_index}")
            return
            
        self.is_capturing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set(f"Capturing from camera {self.camera_index}")
        
        # Start the video thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_capture(self):
        self.is_capturing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        
        if self.cap:
            self.cap.release()
    
    def toggle_recording(self):
        if not self.record_data:
            self.record_data = True
            self.record_button.config(text="Stop Recording")
            self.start_time = datetime.now()
            self.status_var.set("Recording emotion data...")
        else:
            self.record_data = False
            self.record_button.config(text="Start Recording")
            self.save_emotion_data()
            self.status_var.set("Recording stopped. Data saved.")
    
    def save_emotion_data(self):
        if not self.start_time:
            return
            
        # Create a filename with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_folder, f"emotion_data_{timestamp}.csv")
        
        with open(filename, 'w') as f:
            f.write("Emotion,Count\n")
            for emotion, count in self.emotion_counts.items():
                f.write(f"{emotion},{count}\n")
        
        # Reset counts
        for emotion in self.emotion_counts:
            self.emotion_counts[emotion] = 0
    
    def process_video(self):
        last_analysis_time = 0
        analysis_interval = 0.1  # seconds between emotion analyses
        
        while self.is_capturing:
            ret, frame = self.cap.read()
            
            if not ret:
                self.status_var.set("Error: Failed to grab frame")
                break
                
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangle around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Perform emotion analysis only periodically to avoid overloading
            current_time = time.time()
            if current_time - last_analysis_time >= analysis_interval and len(faces) > 0:
                last_analysis_time = current_time
                try:
                    # Perform emotion detection
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(result, list) and len(result) > 0:
                        # Get emotion data
                        emotions = result[0]['emotion']
                        self.current_emotion = result[0]['dominant_emotion']
                        
                        # Get confidence of the dominant emotion
                        confidence = emotions[self.current_emotion]
                        
                        # Update UI in the main thread
                        self.window.after(0, self.update_emotion_display, self.current_emotion, confidence)
                        
                        # Add to history and update counts
                        self.emotion_history.append(self.current_emotion)
                        if self.record_data:
                            self.emotion_counts[self.current_emotion] += 1
                        
                        # Update graph periodically
                        if len(self.emotion_history) % 5 == 0:
                            self.window.after(0, self.update_emotion_graph)
                        
                        # Display emotion on video
                        cv2.putText(frame, f"{self.current_emotion}: {confidence:.1f}%", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 0, 255), 2)
                        
                except Exception as e:
                    print(f"Error in emotion analysis: {e}")
            
            # Convert frame to RGB for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update canvas with new image
            self.window.after(0, self.update_canvas, img_tk)
    
    def update_canvas(self, img_tk):
        self.canvas.img = img_tk  # Keep a reference
        self.canvas.config(width=img_tk.width(), height=img_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    
    def update_emotion_display(self, emotion, confidence):
        self.emotion_label.config(text=emotion.capitalize())
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
    
    def update_emotion_graph(self):
        # Clear the plot
        self.ax.clear()
        
        # Count occurrences of each emotion in history
        if self.emotion_history:
            emotion_counts = {}
            for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
                emotion_counts[emotion] = self.emotion_history.count(emotion)
            
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            # Create bar chart
            bars = self.ax.bar(emotions, counts)
            
            # Color the bars based on emotion
            colors = {'angry': 'red', 'disgust': 'brown', 'fear': 'purple', 
                     'happy': 'green', 'sad': 'blue', 'surprise': 'orange', 'neutral': 'gray'}
            
            for i, bar in enumerate(bars):
                bar.set_color(colors[emotions[i]])
            
            # Add labels and title
            self.ax.set_xlabel('Emotion')
            self.ax.set_ylabel('Count')
            self.ax.set_title('Emotion Distribution')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Update canvas
            self.canvas_graph.draw()
    
    def on_closing(self):
        self.stop_capture()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()