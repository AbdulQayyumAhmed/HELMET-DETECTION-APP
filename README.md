# 👷 Guardian AI: Helmet Detection System

Guardian AI is a professional-grade, high-performance safety surveillance application built with **Streamlit** and **YOLOv8**. It provides real-time PPE (Personal Protective Equipment) compliance monitoring by detecting helmets and identifying safety violations in industrial environments.

![App Screenshot](https://images.unsplash.com/photo-1590486803833-ffc6f11f8fd8?q=80&w=1000&auto=format&fit=crop)

## ✨ Key Features

- **Neural Intelligence:** Powered by YOLOv8 for state-of-the-art object detection.
- **SaaS Dashboard UI:** A premium, Next.js-inspired interface with glassmorphism and dark mode.
- **Real-time Metrics:** Instant feedback on total personnel detected, safe workers, and safety alerts.
- **Side-by-Side Analysis:** Compare original surveillance feeds with neural-annotated results in real-time.
- **Optimized for Monitoring:** Single-screen, landscape layout designed for security control rooms.
- **Automatic Compliance Check:** Categorizes detections into "Safe" and "Violations" based on helmet presence.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Streamlit


### Running the App

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Styling:** Custom CSS (Next.js/React Aesthetic)
- **Computer Vision:** Ultralytics YOLOv8
- **Language:** Python 3.10
- **Image Processing:** PIL (Pillow), NumPy

## 🛡️ Safety Logic

The system identifies violations based on class labels:
- **PPE Safe:** Any detection containing "helmet" (e.g., helmet, blue-helmet).
- **Safety Alerts:** Any detection containing "head" or "no" (e.g., head, no-helmet).

---
*Built with ⚡ by Guardian AI Engineering*
