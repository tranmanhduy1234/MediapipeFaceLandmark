import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# =========================================================================
# CẤU HÌNH CHUNG
# =========================================================================

MODEL_PATH = 'face_landmarker.task'   # File model, để cùng thư mục code
NUM_FACES = 2                          # Nhận diện tối đa 2 khuôn mặt cùng lúc
SAVE_DIR = 'captures'                  # Thư mục lưu ảnh chụp / video
BLENDSHAPE_ALERT_THRESHOLD = 0.5       # Ngưỡng để cảnh báo biểu cảm mạnh (nháy mắt, há miệng...)

os.makedirs(SAVE_DIR, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# =========================================================================
# TRẠNG THÁI HIỂN THỊ (bật/tắt bằng phím tắt trong lúc chạy)
# =========================================================================

class DisplayState:
    def __init__(self):
        self.show_tesselation = True
        self.show_contours = True
        self.show_iris = True
        self.show_blendshape_panel = True
        self.show_head_pose = True
        self.show_fps = True
        self.show_help = True
        self.mirror = True
        self.recording = False

state = DisplayState()

# =========================================================================
# KHỞI TẠO FACE LANDMARKER
# =========================================================================

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=NUM_FACES,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# =========================================================================
# HÀM TIỆN ÍCH
# =========================================================================

def rotation_matrix_to_euler_angles(R):
    """Chuyển ma trận xoay 3x3 thành góc Euler (pitch, yaw, roll) tính bằng độ."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def draw_text_with_background(img, text, org, font_scale=0.55, color=(255, 255, 255),
                               bg_color=(0, 0, 0), thickness=1, alpha=0.5):
    """Vẽ chữ có nền mờ phía sau để dễ đọc hơn trên nền phức tạp."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_blendshape_panel(img, blendshapes, x0=None, y0=10, panel_w=260, bar_h=12, gap=4, top_n=15):
    """Vẽ bảng top-N blendshape mạnh nhất dạng thanh bar ở góc phải màn hình."""
    h, w = img.shape[:2]
    if x0 is None:
        x0 = w - panel_w - 10

    # Bỏ qua "_neutral" khi xếp hạng, sắp xếp giảm dần theo score
    sorted_bs = sorted(blendshapes, key=lambda b: b.score, reverse=True)
    sorted_bs = [b for b in sorted_bs if b.category_name != '_neutral'][:top_n]

    overlay = img.copy()
    panel_h = 24 + len(sorted_bs) * (bar_h + gap + 14)
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x0 + panel_w + 8, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, "BLENDSHAPES (Top {})".format(len(sorted_bs)), (x0, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    y = y0 + 30
    for b in sorted_bs:
        score = b.score
        bar_len = int(score * panel_w)
        color = (0, 255, 0) if score < BLENDSHAPE_ALERT_THRESHOLD else (0, 165, 255)
        cv2.rectangle(img, (x0, y), (x0 + panel_w, y + bar_h), (60, 60, 60), 1)
        cv2.rectangle(img, (x0, y), (x0 + bar_len, y + bar_h), color, -1)
        label = "{}: {:.2f}".format(b.category_name, score)
        cv2.putText(img, label, (x0, y + bar_h + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y += bar_h + gap + 14

    return img

def get_blendshape_value(blendshapes, name):
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0

def draw_help_overlay(img):
    lines = [
        "PHIM TAT:  q=Thoat  s=Chup anh  r=Bat/tat quay video  m=Lat guong",
        "1=Luoi mat  2=Vien mat/moi  3=Con nguoi  4=Bang blendshape",
        "5=Goc dau (pose)  6=FPS  h=An/hien huong dan",
    ]
    y = img.shape[0] - 15 - (len(lines) - 1) * 20
    for line in lines:
        draw_text_with_background(img, line, (20, y), font_scale=0.45, color=(200, 255, 200))
        y += 20

def draw_face_bounding_box(img, face_landmarks_proto, color=(0, 200, 255)):
    """Tính bounding box đơn giản quanh khuôn mặt từ các điểm landmark."""
    h, w = img.shape[:2]
    xs = [lm.x * w for lm in face_landmarks_proto.landmark]
    ys = [lm.y * h for lm in face_landmarks_proto.landmark]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    cv2.rectangle(img, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), color, 2)
    return x_min, y_min, x_max, y_max


# =========================================================================
# VÒNG LẶP CHÍNH
# =========================================================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở Camera.")
        return

    video_writer = None
    prev_time = time.time()
    fps_smooth = 0.0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Đang chạy Face Landmarker (Full Feature)... Nhấn 'q' để thoát, 'h' để xem phím tắt.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Không thể kết nối với Camera.")
                break

            if state.mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            num_faces_detected = len(result.face_landmarks) if result.face_landmarks else 0

            if result.face_landmarks:
                for face_idx, face_landmarks in enumerate(result.face_landmarks):
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    face_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks
                    ])

                    # --- Vẽ lưới / viền / mống mắt (bật/tắt được) ---
                    if state.show_tesselation:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    if state.show_contours:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    if state.show_iris:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )

                    # --- Bounding box + nhãn khuôn mặt (hữu ích khi có nhiều mặt) ---
                    x_min, y_min, x_max, y_max = draw_face_bounding_box(frame, face_landmarks_proto)
                    cv2.putText(frame, "Face #{}".format(face_idx + 1), (x_min - 10, max(y_min - 15, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                    # --- Góc quay đầu (Pitch / Yaw / Roll) + vị trí XYZ ---
                    if state.show_head_pose and result.facial_transformation_matrixes:
                        if face_idx < len(result.facial_transformation_matrixes):
                            matrix = np.array(result.facial_transformation_matrixes[face_idx])
                            x_trans, y_trans, z_trans = matrix[0][3], matrix[1][3], matrix[2][3]
                            pitch, yaw, roll = rotation_matrix_to_euler_angles(matrix[:3, :3])

                            base_y = 30 if face_idx == 0 else 30 + face_idx * 70
                            draw_text_with_background(
                                frame,
                                "F{} Vi tri (cm): X={:.1f} Y={:.1f} Z={:.1f}".format(
                                    face_idx + 1, x_trans, y_trans, z_trans),
                                (20, base_y), color=(0, 255, 0))
                            draw_text_with_background(
                                frame,
                                "F{} Goc dau: Pitch={:.1f} Yaw={:.1f} Roll={:.1f}".format(
                                    face_idx + 1, pitch, yaw, roll),
                                (20, base_y + 25), color=(0, 255, 0))

                    # --- Cảnh báo nháy mắt / há miệng dựa trên blendshape ---
                    if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                        bs = result.face_blendshapes[face_idx]
                        blink_l = get_blendshape_value(bs, 'eyeBlinkLeft')
                        blink_r = get_blendshape_value(bs, 'eyeBlinkRight')
                        jaw_open = get_blendshape_value(bs, 'jawOpen')
                        smile_l = get_blendshape_value(bs, 'mouthSmileLeft')
                        smile_r = get_blendshape_value(bs, 'mouthSmileRight')

                        status_bits = []
                        if blink_l > 0.5 and blink_r > 0.5:
                            status_bits.append("Nham mat")
                        elif blink_l > 0.5:
                            status_bits.append("Nhay mat trai")
                        elif blink_r > 0.5:
                            status_bits.append("Nhay mat phai")
                        if jaw_open > 0.4:
                            status_bits.append("Ha mieng")
                        if (smile_l + smile_r) / 2 > 0.4:
                            status_bits.append("Dang cuoi")

                        if status_bits:
                            cv2.putText(frame, " | ".join(status_bits), (x_min - 10, y_max + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

                    # --- Bảng blendshape chi tiết (chỉ hiển thị cho khuôn mặt đầu tiên để đỡ rối) ---
                    if state.show_blendshape_panel and face_idx == 0 and result.face_blendshapes:
                        draw_blendshape_panel(frame, result.face_blendshapes[0])

            # --- Số khuôn mặt phát hiện được ---
            draw_text_with_background(frame, "So khuon mat: {}".format(num_faces_detected),
                                       (20, frame.shape[0] - 100), color=(255, 255, 0))

            # --- FPS ---
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1 if fps_smooth > 0 else instant_fps
            if state.show_fps:
                draw_text_with_background(frame, "FPS: {:.1f}".format(fps_smooth),
                                           (20, frame.shape[0] - 75), color=(255, 255, 0))

            # --- Trạng thái quay video ---
            if state.recording:
                cv2.circle(frame, (frame.shape[1] - 25, 25), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 70, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                if video_writer is not None:
                    video_writer.write(frame)

            # --- Hướng dẫn phím tắt ---
            if state.show_help:
                draw_help_overlay(frame)

            cv2.imshow('MediaPipe Face Landmarker - Full Feature Demo', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = os.path.join(SAVE_DIR, "capture_{}.png".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S")))
                cv2.imwrite(filename, frame)
                print("Đã lưu ảnh: {}".format(filename))
            elif key == ord('r'):
                state.recording = not state.recording
                if state.recording:
                    video_filename = os.path.join(SAVE_DIR, "video_{}.avi".format(
                        datetime.now().strftime("%Y%m%d_%H%M%S")))
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
                    print("Bắt đầu quay video: {}".format(video_filename))
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("Đã dừng quay video.")
            elif key == ord('m'):
                state.mirror = not state.mirror
            elif key == ord('1'):
                state.show_tesselation = not state.show_tesselation
            elif key == ord('2'):
                state.show_contours = not state.show_contours
            elif key == ord('3'):
                state.show_iris = not state.show_iris
            elif key == ord('4'):
                state.show_blendshape_panel = not state.show_blendshape_panel
            elif key == ord('5'):
                state.show_head_pose = not state.show_head_pose
            elif key == ord('6'):
                state.show_fps = not state.show_fps
            elif key == ord('h'):
                state.show_help = not state.show_help

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()