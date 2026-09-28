import time
import cv2
import numpy as np
from picamera2 import Picamera2
import pigpio
import csv
import os
from pynput import keyboard

CROP_RATIO = 0.2
YAW_PIN = 18
PITCH_PIN = 13
FIRE_PIN = 12

STEP_SIZE = 3.0
MOVE_INTERVAL = 0.1

CENTER_YAW = 50.0
CENTER_PITCH = 90.0

START_YAW = CENTER_YAW
START_PITCH = CENTER_PITCH

FIRE_READY = 90.0
FIRE_SHOOT = 160.0

DATA_DIR = "data_yolo"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
BURST_INTERVAL = 0.2

pi = pigpio.pi()
last_move_time = 0
last_save_time = 0
current_pitch = START_PITCH
current_yaw = START_YAW

pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.space:
            pressed_keys.add("space")
        elif key == keyboard.Key.esc:
            pressed_keys.add("esc")

def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.space:
            pressed_keys.discard("space")
        elif key == keyboard.Key.esc:
            pressed_keys.discard("esc")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def set_servo(pin, angle):
    angle = max(0, min(180, angle))
    pulse = 500 + (angle / 180.0) * 2000.0
    pi.set_servo_pulsewidth(pin, pulse)


set_servo(PITCH_PIN, current_pitch)
set_servo(YAW_PIN, current_yaw)
set_servo(FIRE_PIN, FIRE_READY)


csv_path = os.path.join(DATA_DIR, "data_angles.csv")
file_exists = os.path.isfile(csv_path)
csv_file = open(csv_path, "a", newline="")
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(["filename", "yaw_offset"])

picam = Picamera2()
config = picam.create_preview_configuration(main={"size": (640, 480)})
picam.configure(config)
picam.start()

frame_id = len(os.listdir(IMG_DIR))

r_prev = False

t_prev = False
t_mode = False

print(f"=== Ready: {FIRE_READY} | Shoot: {FIRE_SHOOT} ===")
print("WASD: Move | I: Fire | Space: Save | T: Toggle offset-only mode | R: Reset | ESC: Quit")

try:
    while True:
        raw_frame = picam.capture_array()
        rotated_full = cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = rotated_full.shape[:2]
        start_y = int(h * CROP_RATIO)
        cropped_view = rotated_full[start_y:h, 0:w]
        display_frame = cv2.cvtColor(cropped_view, cv2.COLOR_RGB2BGR)

        current_time = time.time()

        if "t" in pressed_keys and not t_prev:
            t_mode = not t_mode
            print(f"T mode {'ON (motor locked)' if t_mode else 'OFF (motor moves normally)'}")
        t_prev = "t" in pressed_keys

        if current_time - last_move_time > MOVE_INTERVAL:
            if "a" in pressed_keys:
                current_yaw += STEP_SIZE
                if not t_mode:
                    set_servo(YAW_PIN, current_yaw)
                last_move_time = current_time
            elif "d" in pressed_keys:
                current_yaw -= STEP_SIZE
                if not t_mode:
                    set_servo(YAW_PIN, current_yaw)
                last_move_time = current_time

            if "w" in pressed_keys:
                current_pitch += STEP_SIZE
                if not t_mode:
                    set_servo(PITCH_PIN, current_pitch)
                last_move_time = current_time
            elif "s" in pressed_keys:
                current_pitch -= STEP_SIZE
                if not t_mode:
                    set_servo(PITCH_PIN, current_pitch)
                last_move_time = current_time

        if "r" in pressed_keys and not r_prev:
            current_yaw = CENTER_YAW
            current_pitch = CENTER_PITCH
            set_servo(YAW_PIN, current_yaw)
            set_servo(PITCH_PIN, current_pitch)
            print("Reset: position moved to center, offsets set to 0")
        r_prev = "r" in pressed_keys

        if "i" in pressed_keys:
            set_servo(FIRE_PIN, FIRE_SHOOT)
            cv2.putText(
                display_frame,
                "FIRE !",
                (w // 2 - 70, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
        else:
            set_servo(FIRE_PIN, FIRE_READY)

        if "esc" in pressed_keys:
            break

        if "space" in pressed_keys:
            if (current_time - last_save_time) >= BURST_INTERVAL:
                yaw_offset = current_yaw - CENTER_YAW
                filename = f"img_{frame_id:06d}.jpg"
                cv2.imwrite(
                    os.path.join(IMG_DIR, filename),
                    cv2.cvtColor(rotated_full, cv2.COLOR_RGB2BGR),
                )
                writer.writerow([filename, f"{yaw_offset:.1f}"])
                csv_file.flush()
                print(f"Saved: {filename} | Y_Off: {yaw_offset:.1f}")
                frame_id += 1
                last_save_time = current_time
                cv2.rectangle(display_frame, (0, 0), (w, h - start_y), (0, 0, 255), 10)

        yaw_off = current_yaw - CENTER_YAW
        cv2.putText(
            display_frame,
            f"Y_Off: {yaw_off:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        if t_mode:
            cv2.putText(
                display_frame,
                "T MODE (motor locked)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Data Collector", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    picam.stop()
    cv2.destroyAllWindows()
    csv_file.close()
    pi.stop()
    listener.stop()

