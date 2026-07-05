# MediaPipe Face Landmarker Demo & Batch Tools

Dự án này sử dụng mô hình **MediaPipe Face Landmarker** để phát hiện các điểm mốc trên khuôn mặt (468 hoặc 478 điểm), tính toán góc quay đầu (Pitch, Yaw, Roll), đo khoảng cách tương đối (XYZ) và phân tích 52 chỉ số biểu cảm cơ mặt (Blendshapes) tương thích với ARKit.

Dự án hỗ trợ cả hai chế độ:
1. **Nhận diện thời gian thực qua Webcam** (tương tác trực quan, quay phim, chụp ảnh, bật/tắt các lớp đồ họa).
2. **Nhận diện trên ảnh tĩnh** (xuất tọa độ chuẩn hóa và pixel, bounding box, ma trận chuyển đổi ra tệp JSON và vẽ trực quan ra ảnh mới).

---

## 📂 Cấu trúc thư mục chính

* [demo.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/demo.py): Script chạy giao diện nhận diện thời gian thực qua Webcam.
* [Detect.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/Detect.py): Script dòng lệnh trích xuất điểm mốc từ ảnh tĩnh ra tệp JSON.
* [blendshapes.md](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/blendshapes.md): Danh sách chi tiết giải thích ý nghĩa của 52 biểu cảm khuôn mặt.
* [modify.md](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/modify.md): Giải thích chi tiết các trường dữ liệu trong tệp JSON đầu ra.
* `face_landmarker.task`: File mô hình của MediaPipe Face Landmarker (được tải về từ MediaPipe).

---

## 🛠️ Hướng dẫn cài đặt

Dự án yêu cầu Python phiên bản >= 3.8 và một số thư viện cơ bản. Bạn có thể cài đặt bằng lệnh:

```bash
pip install opencv-python mediapipe numpy
```

> [!IMPORTANT]
> Hãy chắc chắn rằng tệp mô hình `face_landmarker.task` nằm cùng thư mục làm việc hiện tại của bạn. Nếu chưa có, bạn có thể tải về từ [Google MediaPipe Face Landmarker Guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models).

---

## 🚀 Cách chạy chương trình

### 1. Chạy nhận diện thời gian thực qua Webcam (Interactive Demo)

Chạy file [demo.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/demo.py) để kích hoạt camera và xem kết quả trực tiếp:

```bash
python3 demo.py
```

Trong lúc chạy, bạn có thể sử dụng các **phím tắt** trên bàn phím để tương tác trực tiếp với giao diện:

* **Điều khiển chung:**
  * `q`: Thoát chương trình.
  * `s`: Chụp ảnh màn hình hiện tại (lưu vào thư mục `captures/`).
  * `r`: Bật/Tắt chế độ quay video (lưu video dưới dạng `.avi` vào thư mục `captures/`).
  * `m`: Bật/Tắt chế độ lật gương (Mirror).
  * `h`: Ẩn/Hiện bảng hướng dẫn phím tắt ở góc dưới màn hình.
* **Bật/Tắt các lớp đồ họa vẽ đè:**
  * `1`: Bật/Tắt lưới bề mặt khuôn mặt (Face Tesselation).
  * `2`: Bật/Tắt đường viền mắt, chân mày, môi (Face Contours).
  * `3`: Bật/Tắt vòng tròn đồng tử mắt (Iris).
  * `4`: Bật/Tắt bảng xếp hạng 15 biểu cảm mạnh nhất (Blendshapes Panel).
  * `5`: Bật/Tắt hiển thị thông tin góc quay đầu (Pitch/Yaw/Roll) và khoảng cách XYZ.
  * `6`: Bật/Tắt chỉ số FPS.

---

### 2. Chạy nhận diện trên ảnh tĩnh (Static Image Batch Tool)

Chạy file [Detect.py](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/Detect.py) để phân tích một bức ảnh và xuất kết quả ra file JSON cùng với ảnh vẽ trực quan:

```bash
python3 Detect.py \
    --image image.jpg \
    --output output.json \
    --model face_landmarker.task \
    --num_faces 1 \
    --draw-output output_drawn.jpg
```

#### Các tham số tùy chọn:
* `--image`: Đường dẫn đến ảnh đầu vào cần xử lý (mặc định: `image.jpg`).
* `--output`: Đường dẫn lưu file kết quả JSON chứa các tọa độ điểm mốc (mặc định: `output.json`).
* `--model`: Đường dẫn đến file mô hình của MediaPipe (mặc định: `face_landmarker.task`).
* `--num_faces`: Số lượng khuôn mặt tối đa muốn nhận diện (mặc định: `1`).
* `--draw-output`: Đường dẫn lưu ảnh kết quả trực quan có vẽ các điểm mốc và hộp bao (mặc định: `output_drawn.jpg`). Nếu không muốn xuất ảnh vẽ đè, hãy truyền chuỗi rỗng `""`.

---

## 📊 Tham khảo thêm

* **Chi tiết cấu trúc JSON xuất ra:** Xem tại file [modify.md](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/modify.md) để biết ý nghĩa từng trường tọa độ pixel, tọa độ chuẩn hóa, bounding box và ma trận xoay 4x4.
* **Chi tiết về 52 Blendshapes:** Xem tại file [blendshapes.md](file:///home/tranmanhduy/Workspace/ptithcm/TTTN/MediaPipeFaceLandmarker/blendshapes.md) để tra cứu tên gọi và nhóm cơ mặt tương ứng.
