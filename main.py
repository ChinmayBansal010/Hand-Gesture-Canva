import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from torchvision import models
from torch import nn
from collections import OrderedDict
from PIL import Image
import math

class HandDrawingApp:
    def __init__(self):
        # ========== Model Setup ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = self.load_class_names("data/class_names.txt")
        self.model = self.load_model("best.pt")
        self.transform = self.get_image_transform()

        # ========== MediaPipe Setup ==========
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # ========== Application State ==========
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_names = ["Blue", "Green", "Red", "Yellow"]
        self.current_color_index = 0
        self.brush_thickness = 10
        self.eraser_thickness = 50
        self.mode = "Idle"
        self.gesture = "None"
        self.prediction = None
        self.prev_x, self.prev_y = 0, 0
        self.canvas = None

        # ========== Video Capture ==========
        self.cap = cv2.VideoCapture(0)

    def load_model(self, path):
        """Loads the PyTorch model's state dictionary into a model instance."""
        try:

            model = models.resnet34()
            num_classes = len(self.class_names)
            if not num_classes > 0:
                print("Error: Class names not loaded. Cannot determine model output size.")
                exit()
            
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            state_dict = torch.load(path, map_location=self.device, weights_only=True)

            if next(iter(state_dict)).startswith('module.'):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] 
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()
            print("Model loaded successfully.")
            return model
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
            exit()
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            exit()

    def load_class_names(self, path):
        """Loads class names from a text file."""
        try:
            with open(path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Class names file not found at {path}")
            return []

    def get_image_transform(self):
        """Defines the image transformation pipeline for the model."""
        is_grayscale = self.model.conv1.weight.shape[1] == 1 if hasattr(self.model, 'conv1') else False # Simplified check
        if is_grayscale:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def predict_canvas(self):
        """Performs prediction on the current canvas content."""
        if self.canvas is None:
            return
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_bin = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
        canvas_rgb = cv2.cvtColor(canvas_bin, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(canvas_rgb)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            if self.class_names:
                self.prediction = self.class_names[pred_idx]
        print(f"Predicted: {self.prediction}")


    def draw_hud(self, frame):
        """Draws the Heads-Up Display on the frame."""
        hud_lines = [
            f"Color: {self.color_names[self.current_color_index]}",
            f"Mode: {self.mode}",
            f"Brush Size: {self.brush_thickness} px",
            f"Gesture: {self.gesture}",
        ]
        if self.prediction:
            hud_lines.append(f"Prediction: {self.prediction}")

        x, y0 = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30

        for i, line in enumerate(hud_lines):
            y = y0 + i * line_height
            cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def fingers_up(self, landmarks):
        """Counts the number of fingers that are up."""
        fingers = []
        fingers.append(landmarks[4].x < landmarks[3].x)
        for tip_id in [8, 12, 16, 20]:
            fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
        return fingers

    def handle_gestures(self, frame, h, w):
        """Processes hand gestures for drawing and controls."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        self.mode = "Idle"
        self.gesture = "None"

        if result.multi_hand_landmarks:
            left_hand, right_hand = None, None
            for hand_landmarks in result.multi_hand_landmarks:
                if hand_landmarks.landmark[0].x < 0.5:
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

            # Left hand controls (brush size, clear canvas)
            if left_hand:
                left_fingers = self.fingers_up(left_hand.landmark)
                if sum(left_fingers) == 5: # All fingers up to clear
                    self.gesture = "Clear Canvas"
                    self.canvas = np.zeros_like(frame)
                    self.prediction = None
                else:
                    self.gesture = "Brush Size"
                    thumb_tip = left_hand.landmark[4]
                    index_tip = left_hand.landmark[8]
                    dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    self.brush_thickness = int(np.interp(dist, [0.05, 0.25], [5, 50]))

            # Right hand controls (drawing, erasing, color selection)
            if right_hand:
                self.mp_draw.draw_landmarks(frame, right_hand, self.mp_hands.HAND_CONNECTIONS)
                right_fingers = self.fingers_up(right_hand.landmark)
                index_x, index_y = int(right_hand.landmark[8].x * w), int(right_hand.landmark[8].y * h)

                # --- GESTURE LOGIC UPDATED ---
                # Gesture 1: Draw with only the index finger up.
                if right_fingers[1] and not any(right_fingers[i] for i in [0, 2, 3, 4]):
                     self.mode = "Drawing"
                     self.gesture = "Pen"
                     if self.prev_x == 0 and self.prev_y == 0:
                         self.prev_x, self.prev_y = index_x, index_y
                     cv2.line(self.canvas, (self.prev_x, self.prev_y), (index_x, index_y), self.colors[self.current_color_index], self.brush_thickness)
                     self.prev_x, self.prev_y = index_x, index_y
                
                # Gesture 2: Select color with index and middle finger up.
                elif right_fingers[1] and right_fingers[2] and not any(right_fingers[i] for i in [0, 3, 4]):
                    self.mode = "Selection"
                    self.gesture = "Select"
                    # --- PALETTE DRAWING LOGIC UPDATED ---
                    # Draw color palette on the TOP RIGHT of the screen.
                    box_size = 50
                    margin = 15
                    for i, color in enumerate(self.colors):
                        x0 = w - (i + 1) * (box_size + margin)
                        x1 = w - (i * (box_size + margin) + margin)
                        y0 = margin
                        y1 = margin + box_size
                        
                        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
                        if x0 < index_x < x1 and y0 < index_y < y1:
                            self.current_color_index = i
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 3) # Highlight selection
                
                # Gesture 3: Erase with a full open palm.
                elif sum(right_fingers) == 5:
                    self.mode = "Erasing"
                    self.gesture = "Eraser"
                    if self.prev_x == 0 and self.prev_y == 0:
                         self.prev_x, self.prev_y = index_x, index_y
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (index_x, index_y), (0,0,0), self.eraser_thickness)
                    self.prev_x, self.prev_y = index_x, index_y
                
                # If no specific gesture is detected, reset the drawing point.
                else:
                    self.prev_x, self.prev_y = 0, 0
            
            # If only the left hand is visible, also reset the drawing point.
            else:
                self.prev_x, self.prev_y = 0, 0
        
        # If no hands are detected, reset the drawing point.
        else:
            self.prev_x, self.prev_y = 0, 0


    def run(self):
        """Main application loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            self.handle_gestures(frame, h, w)

            # Merge canvas with live frame
            frame_with_drawing = cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)

            self.draw_hud(frame_with_drawing)

            # Show frame
            cv2.imshow("Hand Drawing App", frame_with_drawing)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
                self.prediction = None
                
            elif key == ord('s'):
                cv2.imwrite("hand_drawing_output.png", self.canvas)
                
            elif key == ord('p'):
                self.predict_canvas()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()