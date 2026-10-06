import cv2
import numpy as np
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = ROOT / "app" / "data" / "sample_media"
RAW_OUT = ROOT / "app" / "data" / "processed_media" / "temp_road_surv_raw.mp4"
FINAL_OUT = SAMPLE_DIR / "road_surveillance_traffic.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 25
DURATION_SEC = 22
TOTAL_FRAMES = int(FPS * DURATION_SEC)

HORIZON_Y = 160
ROAD_TOP_LEFT = 460
ROAD_TOP_RIGHT = 820
ROAD_BOT_LEFT = 40
ROAD_BOT_RIGHT = 1240

# Lanes: 0=Left, 1=Middle, 2=Right
LANES = [
    {"name": "Lane 1 (Passing)", "top_x": 520, "bot_x": 240},
    {"name": "Lane 2 (Cruising)", "top_x": 640, "bot_x": 640},
    {"name": "Lane 3 (Heavy)", "top_x": 760, "bot_x": 1040},
]

# Vehicle configs with start time and durations
vehicles_meta = [
    {
        "file": "sample_car_4.jpg",
        "crop_box": (0.35, 0.85, 0.15, 0.85),
        "plate": "KL 02 BM 4659",
        "lane": 1,
        "start_sec": 0.5,
        "duration": 4.2,
        "type": "Executive Sedan"
    },
    {
        "file": "sample_car_1.jpg",
        "crop_box": (0.28, 0.88, 0.10, 0.90),
        "plate": "AP 29 AN 0074",
        "lane": 0,
        "start_sec": 4.2,
        "duration": 4.2,
        "type": "Hatchback"
    },
    {
        "file": "sample_tempo_3.jpg",
        "crop_box": (0.20, 0.82, 0.08, 0.92),
        "plate": "JH 05 AW 2117",
        "lane": 2,
        "start_sec": 8.0,
        "duration": 4.5,
        "type": "Commercial Van"
    },
    {
        "file": "sample_car_2.jpg",
        "crop_box": (0.40, 0.88, 0.12, 0.88),
        "plate": "MP 42 MG 2246",
        "lane": 1,
        "start_sec": 12.0,
        "duration": 4.2,
        "type": "Compact Sedan"
    },
    {
        "file": "sample_car_4.jpg",
        "crop_box": (0.35, 0.85, 0.15, 0.85),
        "plate": "HR 26 DA 2330",
        "lane": 0,
        "start_sec": 16.0,
        "duration": 4.2,
        "type": "Premium Sedan"
    }
]

# Load and prepare car crops
loaded_cars = []
for v in vehicles_meta:
    img = cv2.imread(str(SAMPLE_DIR / v["file"]))
    if img is not None:
        ih, iw = img.shape[:2]
        y1, y2, x1, x2 = v["crop_box"]
        crop = img[int(ih*y1):int(ih*y2), int(iw*x1):int(iw*x2)]
        v["crop"] = crop
        loaded_cars.append(v)

print(f"Loaded {len(loaded_cars)} vehicle models.")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(RAW_OUT), fourcc, FPS, (WIDTH, HEIGHT))

for frame_idx in range(TOTAL_FRAMES):
    t_sec = frame_idx / float(FPS)
    
    # 1. Base Road Frame
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Sky
    frame[0:HORIZON_Y, :] = (215, 195, 175)
    for x in range(WIDTH):
        hill = int(14 * np.sin(x * 0.015) + 8 * np.cos(x * 0.04))
        frame[HORIZON_Y - 35 - hill:HORIZON_Y, x] = (120, 135, 105)

    # Roadside green grass verges
    pts_left_grass = np.array([[0, HORIZON_Y], [ROAD_TOP_LEFT, HORIZON_Y], [ROAD_BOT_LEFT, HEIGHT], [0, HEIGHT]], np.int32)
    pts_right_grass = np.array([[ROAD_TOP_RIGHT, HORIZON_Y], [WIDTH, HORIZON_Y], [WIDTH, HEIGHT], [ROAD_BOT_RIGHT, HEIGHT]], np.int32)
    cv2.fillPoly(frame, [pts_left_grass], (42, 92, 48))
    cv2.fillPoly(frame, [pts_right_grass], (42, 92, 48))

    # Dark Asphalt roadway
    pts_road = np.array([[ROAD_TOP_LEFT, HORIZON_Y], [ROAD_TOP_RIGHT, HORIZON_Y], [ROAD_BOT_RIGHT, HEIGHT], [ROAD_BOT_LEFT, HEIGHT]], np.int32)
    cv2.fillPoly(frame, [pts_road], (45, 48, 52))

    # Yellow shoulder boundary lines
    cv2.line(frame, (ROAD_TOP_LEFT + 8, HORIZON_Y), (ROAD_BOT_LEFT + 25, HEIGHT), (40, 205, 245), 3, cv2.LINE_AA)
    cv2.line(frame, (ROAD_TOP_RIGHT - 8, HORIZON_Y), (ROAD_BOT_RIGHT - 25, HEIGHT), (40, 205, 245), 3, cv2.LINE_AA)

    # Dashed lane divider lines with road motion effect
    div1_top = int(ROAD_TOP_LEFT + (ROAD_TOP_RIGHT - ROAD_TOP_LEFT) * 0.333)
    div1_bot = int(ROAD_BOT_LEFT + (ROAD_BOT_RIGHT - ROAD_BOT_LEFT) * 0.333)
    div2_top = int(ROAD_TOP_LEFT + (ROAD_TOP_RIGHT - ROAD_TOP_LEFT) * 0.667)
    div2_bot = int(ROAD_BOT_LEFT + (ROAD_BOT_RIGHT - ROAD_BOT_LEFT) * 0.667)

    dash_speed = (frame_idx * 2.0) % 50
    for s_step in range(0, 100, 10):
        prog1 = (s_step + dash_speed) / 100.0
        prog2 = (s_step + dash_speed + 5) / 100.0
        if prog1 <= 1.0 and prog2 <= 1.0:
            p1 = prog1 ** 1.8
            p2 = prog2 ** 1.8
            y_a = int(HORIZON_Y + (HEIGHT - HORIZON_Y) * p1)
            y_b = int(HORIZON_Y + (HEIGHT - HORIZON_Y) * p2)
            
            x1_a = int(div1_top + (div1_bot - div1_top) * p1)
            x1_b = int(div1_top + (div1_bot - div1_top) * p2)
            cv2.line(frame, (x1_a, y_a), (x1_b, y_b), (245, 245, 245), max(1, int(3 * p1)), cv2.LINE_AA)

            x2_a = int(div2_top + (div2_bot - div2_top) * p1)
            x2_b = int(div2_top + (div2_bot - div2_top) * p2)
            cv2.line(frame, (x2_a, y_a), (x2_b, y_b), (245, 245, 245), max(1, int(3 * p1)), cv2.LINE_AA)

    # 2. Render Active Vehicles
    active_in_frame = []
    for v in loaded_cars:
        if v["start_sec"] <= t_sec <= (v["start_sec"] + v["duration"]):
            prog = (t_sec - v["start_sec"]) / float(v["duration"])
            ease_prog = prog ** 1.35
            
            lane_info = LANES[v["lane"]]
            car_x = int(lane_info["top_x"] + (lane_info["bot_x"] - lane_info["top_x"]) * ease_prog)
            car_y = int(HORIZON_Y + 15 + (HEIGHT + 140 - (HORIZON_Y + 15)) * ease_prog)
            
            # Smooth scaling from 0.22 at distance to 0.82 in foreground
            car_scale = 0.22 + 0.60 * ease_prog
            active_in_frame.append((car_y, car_x, car_scale, v, ease_prog))

    active_in_frame.sort(key=lambda item: item[0])

    for car_y, car_x, car_scale, v, ease_prog in active_in_frame:
        crop = v["crop"]
        ch, cw = crop.shape[:2]
        cur_w = int(cw * car_scale * 0.32)
        cur_h = int(ch * car_scale * 0.32)
        
        if cur_w < 10 or cur_h < 10:
            continue

        resized = cv2.resize(crop, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
        
        pos_x1 = car_x - cur_w // 2
        pos_y1 = car_y - cur_h
        pos_x2 = pos_x1 + cur_w
        pos_y2 = pos_y1 + cur_h

        # Shadow
        shadow_cy = min(HEIGHT - 4, pos_y2 - int(cur_h * 0.07))
        shadow_rx = int(cur_w * 0.48)
        shadow_ry = max(4, int(cur_h * 0.12))
        if 0 <= car_x < WIDTH and 0 <= shadow_cy < HEIGHT:
            cv2.ellipse(frame, (car_x, shadow_cy), (shadow_rx, shadow_ry), 0, 0, 360, (20, 22, 26), -1)

        # Blit car
        src_x1 = max(0, -pos_x1)
        src_y1 = max(0, -pos_y1)
        src_x2 = cur_w - max(0, pos_x2 - WIDTH)
        src_y2 = cur_h - max(0, pos_y2 - HEIGHT)

        dst_x1 = max(0, pos_x1)
        dst_y1 = max(0, pos_y1)
        dst_x2 = min(WIDTH, pos_x2)
        dst_y2 = min(HEIGHT, pos_y2)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1 and src_x2 > src_x1 and src_y2 > src_y1:
            frame[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

    # 3. Clean CCTV Telemetry in top corner
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (15, 15), (520, 50), (15, 20, 26), -1)
    cv2.addWeighted(hud_bg, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (15, 15), (520, 50), (0, 220, 130), 1)

    # Blinking REC dot
    if (frame_idx // 12) % 2 == 0:
        cv2.circle(frame, (32, 32), 6, (0, 0, 240), -1)
    
    cv2.putText(frame, "LIVE SURVEILLANCE  CAM-04 (HIGHWAY TOLLWAY)", (48, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Live Clock overlay in top right
    time_bg = frame.copy()
    cv2.rectangle(time_bg, (WIDTH - 340, 15), (WIDTH - 15, 50), (15, 20, 26), -1)
    cv2.addWeighted(time_bg, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (WIDTH - 340, 15), (WIDTH - 15, 50), (0, 220, 130), 1)
    
    sec_int = int(t_sec)
    time_str = f"2026-10-06 17:15:{sec_int:02d} | 25 FPS"
    cv2.putText(frame, time_str, (WIDTH - 325, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 130), 1, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"Raw video generated at {RAW_OUT}.")

ffmpeg_cmd = [
    "/opt/homebrew/bin/ffmpeg",
    "-y",
    "-i", str(RAW_OUT),
    "-c:v", "libx264",
    "-profile:v", "high",
    "-level", "4.0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-b:v", "3500k",
    str(FINAL_OUT)
]
subprocess.run(ffmpeg_cmd, check=True)
print(f"Final browser-compatible road surveillance video ready at {FINAL_OUT}!")
