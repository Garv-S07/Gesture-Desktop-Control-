from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fixed gesture map
BASE_GESTURES = {
    "move": {"name": "move", "action": "Move Cursor"},
    "left_click": {"name": "left_click", "action": "Left Mouse Click"},
    "right_click": {"name": "right_click", "action": "Right Mouse Click"},
    "scroll_up": {"name": "scroll_up", "action": "Scroll Up"},
    "scroll_down": {"name": "scroll_down", "action": "Scroll Down"},
    "maximize": {"name": "maximize", "action": "Maximize Window"},
    "minimize": {"name": "minimize", "action": "Minimize Window"}
}

class RecordCommand(BaseModel):
    user_id: str
    gesture_name: str

def load_data():
    if not os.path.exists('usergestures.json'):
        return {}
    with open('usergestures.json','r') as f:
        try: return json.load(f)
        except json.JSONDecodeError: return {}

def save_data(data):
    with open('usergestures.json','w') as f:
        json.dump(data, f, indent=4)

@app.post("/login/{user_id}")
def login_user(user_id: str):
    data = load_data()
    
    if user_id not in data:
        data[user_id] = BASE_GESTURES.copy()
        save_data(data)
        
        # Copy base CSV data
        base_csv_path = "model/base_data.csv"
        user_csv_path = f"model/{user_id}_data.csv"
        if os.path.exists(base_csv_path):
            shutil.copy(base_csv_path, user_csv_path)
            
        # Tell OpenCV to train the new model
        with open("control.json", "w") as f:
            json.dump({"record": False, "retrain": True, "user_id": user_id}, f)

    # If user exists, just load their active model
    else:
        with open("control.json", "w") as f:
            json.dump({"record": False, "retrain": True, "user_id": user_id}, f)

    return {"message": "Login successful", "user_id": user_id}

@app.get("/user/{user_id}/gestures")
def get_user_gestures(user_id: str):
    data = load_data()
    if user_id in data:
        return data[user_id]
    raise HTTPException(status_code=404, detail="User not found.")

@app.post("/command/record")
def send_record_command(cmd: RecordCommand):
    # Send record command to OpenCV
    command_data = {
        "record": True,
        "retrain": False,
        "user_id": cmd.user_id,
        "gesture_name": cmd.gesture_name
    }
    with open("control.json", "w") as f:
        json.dump(command_data, f)
        
    return {"message": "Recording initiated"}