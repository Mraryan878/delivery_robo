# Delivery Robot

Small project that runs a delivery-robot ML prototype.

Contents
- `server.py`: Flask/stream server.
- `ai_model.py`: model wrapper.
- `requirements.txt`: Python dependencies.
- `yolov8n.pt`: model weights (not committed if large).

Quick start
1. Create a virtual environment and install requirements:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the server:

```bash
python server.py
```

Repository
https://github.com/Mraryan878/delivery_robo

Notes
- `captured_images/` is ignored by git to avoid large files.
- Add an SSH key or use `gh auth login --web` for authenticated pushes.
