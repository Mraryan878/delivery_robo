# config.py - Server Configuration

# Network Settings
ESP32_CAM_IP = "10.105.131.124"  # Set this to your ESP32-CAM's IP
ESP32_DEV_IP = "192.168.1.101"   # Set this to your ESP32 Dev Board's IP

# Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# Motor Settings
MOTOR_URL = f"http://{ESP32_DEV_IP}:80/command"
MOTOR_TIMEOUT = 0.5  # seconds

# AI Settings
AI_MODEL_NAME = "yolov8n.pt"
AI_CONFIDENCE_THRESHOLD = 0.5

# Camera Settings (for ESP32-CAM)
CAMERA_FRAME_SIZE = "QVGA"  # QVGA, QQVGA, etc.
CAMERA_JPEG_QUALITY = 12