import os, datetime, cv2, numpy as np

class ComplianceLogger:
    def __init__(self, log_file="logs/violations.log", screenshot_dir="logs/violations"):
        self.log_file = log_file
        self.ss_dir = screenshot_dir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
        self.log_messages = []   # for live window

    def log(self, label, confidence, frame=None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] {label.upper()} – {confidence*100:.1f}%"
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        self.log_messages.append(msg)
        if frame is not None:
            fname = f"{self.ss_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
        # Keep only last 15 messages for display
        if len(self.log_messages) > 15:
            self.log_messages.pop(0)

    def show_log_window(self):
        # Create a separate window with recent logs
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        y0 = 30
        for i, msg in enumerate(self.log_messages[-10:]):
            cv2.putText(canvas, msg, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Violation Log", canvas)