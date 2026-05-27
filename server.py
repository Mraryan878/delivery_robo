from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
import requests
import time
import io
import os
from ultralytics import YOLO

# ========== Configuration ==========
ESP32_DEV_IP = "10.133.196.80"
ESP32_DEV_API = f"http://{ESP32_DEV_IP}"
MOTOR_URL = f"{ESP32_DEV_API}/api/command"
DISTANCE_URL = f"{ESP32_DEV_API}/api/distance"

# ========== Search Settings ==========
ROTATION_DURATION = 0.15      # Rotate for 0.15 seconds (~30 degrees)
STOP_DURATION = 5.0           # STOP completely for 5 seconds

# ========== Detection Settings ==========
TARGET_OBJECT = "bottle"
CONFIDENCE_THRESHOLD = 0.35

# ========== Initialize ==========
app = Flask(__name__)
CORS(app)

# State variables
state = "SEEKING"  # SEEKING, FOLLOWING, RETURNING
search_step = 0
path_memory = []
frame_count = 0
last_image_bytes = None

# Load AI
print(f"[INFO] Loading YOLO model (target: {TARGET_OBJECT})...")
model = YOLO('yolov8n.pt')
print("[INFO] Model loaded!")

# ========== Helper Functions ==========
def send_motor_cmd(cmd, duration=None):
    """Send motor command, optionally auto-stop after duration"""
    try:
        requests.post(MOTOR_URL, data={'action': cmd}, timeout=0.3)
        print(f"📤 Motor: {cmd}")
        if duration:
            time.sleep(duration)
            requests.post(MOTOR_URL, data={'action': 'STP'}, timeout=0.3)
            print(f"📤 Motor: STOP after {duration}s")
        return True
    except:
        print(f"⚠️ Motor failed: {cmd}")
        return False

def stop_motors():
    try:
        requests.post(MOTOR_URL, data={'action': 'STP'}, timeout=0.3)
    except:
        pass

def get_distance():
    try:
        r = requests.get(DISTANCE_URL, timeout=0.5)
        if r.status_code == 200:
            return int(r.text)
    except:
        pass
    return 999

def detect_object(img):
    results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            if cls == TARGET_OBJECT:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    'center': (x1 + x2) / 2,
                    'width': x2 - x1,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
    return detections

# ========== HTML ==========
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Delivery Robot</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; text-align: center; padding: 20px; }
        img { max-width: 640px; border: 2px solid #00ff88; border-radius: 10px; }
        .info { background: #0f0f23; padding: 15px; margin: 20px auto; max-width: 500px; border-radius: 10px; }
        button { background: #00ff88; color: #1a1a2e; padding: 12px 24px; font-size: 16px; margin: 5px; border: none; border-radius: 8px; cursor: pointer; }
        .return { background: #ff8800; color: white; }
    </style>
</head>
<body>
    <h1>🤖 Delivery Robot</h1>
    <img id="cam" src="/api/last_image">
    <div class="info">
        <p>State: <span id="state">---</span></p>
        <p>Step: <span id="step">0</span>/12</p>
        <p>Detections: <span id="det">0</span></p>
    </div>
    <button onclick="start()">🚚 START DELIVERY</button>
    <button onclick="returnHome()" class="return">🏠 RETURN HOME</button>
    <script>
        function refresh() {
            document.getElementById('cam').src = '/api/last_image?t='+Date.now();
            fetch('/api/stats').then(r=>r.json()).then(d=>{
                document.getElementById('state').innerText = d.state;
                document.getElementById('step').innerText = d.step;
                document.getElementById('det').innerText = d.detections;
            });
        }
        function start() { fetch('/api/start', {method:'POST'}); }
        function returnHome() { fetch('/api/return', {method:'POST'}); }
        setInterval(refresh, 500);
        refresh();
    </script>
</body>
</html>
'''

# ========== Routes ==========
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/last_image')
def last_image():
    global last_image_bytes
    if last_image_bytes:
        return send_file(io.BytesIO(last_image_bytes), mimetype='image/jpeg')
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    _, img = cv2.imencode('.jpg', blank)
    return send_file(io.BytesIO(img.tobytes()), mimetype='image/jpeg')

@app.route('/api/stats')
def stats():
    global state, search_step, last_detections
    return jsonify({
        'state': state,
        'step': search_step,
        'detections': last_detections
    })

@app.route('/api/start', methods=['POST'])
def start():
    global state, search_step, path_memory
    state = "SEEKING"
    search_step = 0
    path_memory = []
    print("🚚 DELIVERY STARTED!")
    return jsonify({'status': 'ok'})

@app.route('/api/return', methods=['POST'])
def return_to_start():
    global state
    state = "RETURNING"
    print("🏠 RETURNING HOME...")
    return jsonify({'status': 'ok'})

@app.route('/upload', methods=['POST'])
def upload():
    global frame_count, last_image_bytes, last_detections, state, search_step, path_memory
    
    try:
        # Get image
        img_bytes = request.get_data()
        if not img_bytes:
            return jsonify({'error': 'No image'}), 400
        
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Store image for web
        _, img_encoded = cv2.imencode('.jpg', img)
        last_image_bytes = img_encoded.tobytes()
        
        # Check distance (obstacle avoidance)
        distance = get_distance()
        if 0 < distance < 25:
            print(f"⚠️ OBSTACLE at {distance}cm!")
            stop_motors()
            return jsonify({'command': 'STOP'}), 200
        
        # Run detection
        detections = detect_object(img)
        last_detections = len(detections)
        
        # Annotate image
        annotated = img.copy()
        for d in detections:
            cv2.rectangle(annotated, (d['x1'], d['y1']), (d['x2'], d['y2']), (0,255,0), 2)
        cv2.putText(annotated, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        _, annotated_enc = cv2.imencode('.jpg', annotated)
        global last_annotated_bytes
        last_annotated_bytes = annotated_enc.tobytes()
        
        frame_count += 1
        
        # ========== STATE MACHINE ==========
        
        if state == "SEEKING":
            if detections:
                print(f"🎯 TARGET FOUND at step {search_step}! Switching to FOLLOW.")
                state = "FOLLOWING"
                stop_motors()
                return jsonify({'command': 'FOUND'}), 200
            else:
                # Rotate, then STOP, then wait for next frame
                search_step += 1
                if search_step > 12:
                    search_step = 1  # Reset after full circle
                
                # Rotate for 0.15s (~30 degrees)
                send_motor_cmd('RGT', ROTATION_DURATION)
                print(f"🔍 SEEKING: Step {search_step}/12 - Rotated, now STOPPING for {STOP_DURATION}s")
                
                # CRITICAL: Wait for robot to stabilize before next frame
                time.sleep(STOP_DURATION)
                
                return jsonify({'command': 'ROTATE', 'step': search_step}), 200
        
        elif state == "FOLLOWING":
            if not detections:
                print("❌ TARGET LOST! Returning to SEEK.")
                state = "SEEKING"
                stop_motors()
                return jsonify({'command': 'LOST'}), 200
            
            # Follow the target
            best = detections[0]
            center = best['center']
            frame_center = img.shape[1] / 2
            
            if abs(center - frame_center) < 40:
                cmd = 'FWD'
                path_memory.append('FWD')
                send_motor_cmd(cmd, 0.2)
                print(f"🎯 FOLLOW: Moving FORWARD")
            elif center < frame_center:
                cmd = 'LFT'
                send_motor_cmd(cmd, 0.15)
                print(f"🎯 FOLLOW: Turning LEFT")
            else:
                cmd = 'RGT'
                send_motor_cmd(cmd, 0.15)
                print(f"🎯 FOLLOW: Turning RIGHT")
            
            time.sleep(0.3)  # Stabilize
            return jsonify({'command': cmd}), 200
        
        elif state == "RETURNING":
            if path_memory:
                last_cmd = path_memory.pop()
                rev = {'FWD':'FWD', 'LFT':'RGT', 'RGT':'LFT'}.get(last_cmd, 'STP')
                send_motor_cmd(rev, 0.2)
                print(f"🔙 RETURN: {rev} | Steps left: {len(path_memory)}")
                time.sleep(0.3)
            else:
                print("🏁 MISSION COMPLETE!")
                state = "SEEKING"
                stop_motors()
            return jsonify({'command': rev if path_memory else 'STP'}), 200
        
        return jsonify({'command': 'IDLE'}), 200
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

# Global for annotated image
last_annotated_bytes = None

@app.route('/api/last_image_annotated')
def last_image_annotated():
    global last_annotated_bytes
    if last_annotated_bytes:
        return send_file(io.BytesIO(last_annotated_bytes), mimetype='image/jpeg')
    return last_image()

if __name__ == '__main__':
    print("=" * 50)
    print("🤖 DELIVERY ROBOT - SIMPLE STOP & ANALYZE")
    print("=" * 50)
    print(f"📍 Motor API: {MOTOR_URL}")
    print(f"⏱️  Rotation: {ROTATION_DURATION}s then STOP")
    print(f"📷 Stop Duration: {STOP_DURATION}s")
    print("\n🌐 OPEN: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)