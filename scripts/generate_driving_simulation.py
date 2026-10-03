"""
Generates a realistic synthetic driving dashcam video with approaching traffic signs.
Used to test continuous-time vehicle video stream detection, tracking, and ADAS HUD.
"""

import os
import cv2
import numpy as np


def generate_driving_video(output_path: str = "data/samples/simulated_driving.mp4", duration_sec: int = 10, fps: int = 30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 360
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    horizon_y = int(height * 0.45)

    # Road geometry
    road_top_left = (int(width * 0.46), horizon_y)
    road_top_right = (int(width * 0.54), horizon_y)
    road_bot_left = (int(width * 0.05), height)
    road_bot_right = (int(width * 0.95), height)

    def draw_speed_50(size):
        """Draw a 50 km/h speed limit sign disc."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Transparent/background
        cv2.circle(img, (size//2, size//2), size//2 - 1, (40, 40, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (size//2, size//2), int(size * 0.38), (250, 250, 250), -1, cv2.LINE_AA)
        font_s = max(0.25, size / 65.0)
        thick = max(1, int(size / 30.0))
        (tw, th), _ = cv2.getTextSize("50", cv2.FONT_HERSHEY_SIMPLEX, font_s, thick)
        cv2.putText(img, "50", (size//2 - tw//2, size//2 + th//2), cv2.FONT_HERSHEY_SIMPLEX, font_s, (20, 20, 20), thick, cv2.LINE_AA)
        return img

    def draw_danger_sign(size):
        """Draw a triangular road work / danger sign."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        pt1 = (size//2, 2)
        pt2 = (2, size - 3)
        pt3 = (size - 3, size - 3)
        cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, (40, 40, 230), -1, cv2.LINE_AA)
        # Inner white triangle
        ipt1 = (size//2, int(size * 0.22))
        ipt2 = (int(size * 0.18), size - int(size * 0.12))
        ipt3 = (size - int(size * 0.18), size - int(size * 0.12))
        cv2.drawContours(img, [np.array([ipt1, ipt2, ipt3])], 0, (250, 250, 250), -1, cv2.LINE_AA)
        # Exclamation / symbol
        thick = max(1, int(size / 25.0))
        cv2.line(img, (size//2, int(size * 0.40)), (size//2, int(size * 0.65)), (20, 20, 20), thick, cv2.LINE_AA)
        cv2.circle(img, (size//2, int(size * 0.78)), max(1, thick), (20, 20, 20), -1)
        return img

    def draw_keep_right(size):
        """Draw a blue mandatory keep right sign."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(img, (size//2, size//2), size//2 - 1, (230, 130, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (size//2, size//2), int(size * 0.45), (250, 250, 250), 2, cv2.LINE_AA)
        # Arrow pointing down-right
        p1 = (int(size * 0.35), int(size * 0.35))
        p2 = (int(size * 0.65), int(size * 0.65))
        thick = max(1, int(size / 16.0))
        cv2.arrowedLine(img, p1, p2, (255, 255, 255), thick, tipLength=0.4)
        return img

    # Sign schedule: (start_frame, end_frame, sign_generator_func)
    sign_schedule = [
        (15, 95, draw_speed_50),
        (110, 190, draw_danger_sign),
        (205, 285, draw_keep_right)
    ]

    print(f"Generating synthetic driving simulation ({total_frames} frames)...")

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky & Ground
        frame[0:horizon_y, :] = (210, 180, 140)   # Day sky
        frame[horizon_y:height, :] = (80, 130, 70) # Grass shoulder

        # 2. Road surface (perspective polygon)
        road_pts = np.array([road_top_left, road_top_right, road_bot_right, road_bot_left])
        cv2.fillPoly(frame, [road_pts], (55, 55, 55)) # Dark asphalt

        # 3. Dashed center lane divider
        lane_progress = (f * 6) % 40
        for y_dash in range(horizon_y + 10, height, 40):
            yd = y_dash + lane_progress
            if yd >= height:
                continue
            t = (yd - horizon_y) / float(height - horizon_y)
            cx = int(width * 0.50)
            dash_len = int(8 + t * 24)
            dash_thick = max(1, int(1 + t * 4))
            cv2.line(frame, (cx, yd), (cx, min(height, yd + dash_len)), (240, 240, 240), dash_thick, cv2.LINE_AA)

        # 4. Approaching traffic signs
        for (start_f, end_f, sign_fn) in sign_schedule:
            if start_f <= f <= end_f:
                progress = (f - start_f) / float(end_f - start_f) # 0.0 to 1.0
                # Scale from 16px to 95px
                sign_size = int(16 + (progress ** 2.2) * 85)
                # Sign approaches on the right roadside
                sign_x = int((width * 0.58) + progress * (width * 0.32))
                sign_y = int(horizon_y + progress * (height * 0.32) - sign_size // 2)

                # Post pole
                pole_top = (sign_x + sign_size // 2, sign_y + sign_size)
                pole_bot = (sign_x + sign_size // 2, min(height - 10, sign_y + sign_size + int(30 + progress * 70)))
                cv2.line(frame, pole_top, pole_bot, (120, 120, 120), max(1, int(sign_size / 24)), cv2.LINE_AA)

                # Draw sign patch
                patch = sign_fn(sign_size)
                sy1 = max(0, sign_y)
                sy2 = min(height, sign_y + sign_size)
                sx1 = max(0, sign_x)
                sx2 = min(width, sign_x + sign_size)

                ph = sy2 - sy1
                pw = sx2 - sx1
                if ph > 0 and pw > 0:
                    sub_patch = patch[0:ph, 0:pw]
                    mask = (sub_patch > 0).any(axis=2)
                    frame[sy1:sy2, sx1:sx2][mask] = sub_patch[mask]

        writer.write(frame)

    writer.release()
    print(f"[Success] Generated driving simulation video at: {output_path}")


if __name__ == "__main__":
    generate_driving_video()
