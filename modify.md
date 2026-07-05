# Hướng dẫn sử dụng & Cấu trúc dữ liệu JSON từ Face Landmarker

Tài liệu này hướng dẫn cách chạy script `Detect.py` để trích xuất tọa độ điểm mốc và bounding box của khuôn mặt từ một ảnh tĩnh, đồng thời mô tả chi tiết cấu trúc dữ liệu của file JSON kết quả.

---

## 1. Cách chạy script Detect.py

Bạn có thể chạy script `Detect.py` bằng Python với các tham số cấu hình linh hoạt:

```bash
python3 Detect.py \
    --image image.jpg \
    --output output.json \
    --model face_landmarker.task \
    --num_faces 1 \
    --draw-output output_drawn.jpg
```

### Các tham số:
* `--image`: Đường dẫn đến ảnh đầu vào cần xử lý (mặc định: `image.jpg`).
* `--output`: Đường dẫn lưu file kết quả JSON (mặc định: `output.json`).
* `--model`: Đường dẫn đến file mô hình của MediaPipe (mặc định: `face_landmarker.task`).
* `--num_faces`: Số lượng khuôn mặt tối đa muốn nhận diện (mặc định: `1`).
* `--draw-output`: Đường dẫn lưu ảnh kết quả trực quan có vẽ các điểm mốc và hộp bao (mặc định: `output_drawn.jpg`). Nếu không muốn vẽ, hãy truyền chuỗi rỗng `""`.

---

## 2. Cấu trúc dữ liệu file JSON đầu ra

Dưới đây là cấu trúc định dạng dữ liệu (JSON Schema) chi tiết được tạo ra bởi script:

```json
{
    "image_info": {
        "path": "/đường/dẫn/tuyệt/đối/tới/ảnh/image.jpg",
        "width": 153,
        "height": 178
    },
    "faces": [
        {
            "face_index": 0,
            "landmarks_normalized": [
                {
                    "index": 0,
                    "x": 0.5076854825019836,
                    "y": 0.7135779857635498,
                    "z": -0.026174021884799004
                }
                // ... (tổng cộng 468 hoặc 478 điểm mốc)
            ],
            "landmarks_pixel": [
                {
                    "index": 0,
                    "x": 77,
                    "y": 127,
                    "z": -4.004625348374248
                }
                // ... (tổng cộng 468 hoặc 478 điểm mốc quy đổi ra pixel thực tế)
            ],
            "bounding_box_normalized": {
                "xmin": 0.23772025108337402,
                "ymin": 0.18899041414260864,
                "xmax": 1.021912693977356,
                "ymax": 0.9140231013298035,
                "width": 0.7841924428939819,
                "height": 0.7250326871871948
            },
            "bounding_box_pixel": {
                "xmin": 36,
                "ymin": 33,
                "xmax": 156,
                "ymax": 162,
                "width": 120,
                "height": 129
            },
            "blendshapes": [
                {
                    "category_name": "_neutral",
                    "score": 0.0983271
                }
                // ... (52 loại biểu cảm khuôn mặt cơ bản)
            ],
            "facial_transformation_matrix": [
                [ 0.9388, -0.1294, -0.3189,  1.7860 ],
                [ -0.0053,  0.9209, -0.3895,  0.5889 ],
                [ 0.3441,  0.3674,  0.8639, -20.1521 ],
                [ 0.0,     0.0,     0.0,      1.0 ]
            ]
        }
    ]
}
```

---

## 3. Giải thích chi tiết các trường dữ liệu

| Trường | Kiểu dữ liệu | Ý nghĩa |
| :--- | :--- | :--- |
| **`image_info`** | `Object` | Thông tin chung về tệp ảnh đầu vào để tham chiếu kích thước gốc. |
| **`image_info.path`** | `String` | Đường dẫn tuyệt đối dẫn đến tệp ảnh đã xử lý. |
| **`image_info.width`** | `Integer` | Chiều rộng (width) của ảnh đầu vào tính bằng pixel. |
| **`image_info.height`** | `Integer` | Chiều cao (height) của ảnh đầu vào tính bằng pixel. |
| **`faces`** | `Array` | Mảng chứa thông tin của tất cả các khuôn mặt được phát hiện. |
| **`faces[i].face_index`** | `Integer` | Chỉ số thứ tự của khuôn mặt (từ `0` đến `n-1`). |
| **`faces[i].landmarks_normalized`** | `Array` | Danh sách các điểm mốc chuẩn hóa có tọa độ `x, y, z` từ `0.0` đến `1.0` so với góc trên bên trái ảnh. |
| **`faces[i].landmarks_pixel`** | `Array` | Tọa độ điểm mốc quy đổi về kích thước điểm ảnh thực tế (`x = x_norm * width`, `y = y_norm * height`). Tọa độ `z` được scale tương đối theo trục x. |
| **`faces[i].bounding_box_normalized`** | `Object` | Hộp bao quanh tất cả các điểm mốc khuôn mặt dưới dạng tọa độ chuẩn hóa (`xmin`, `ymin`, `xmax`, `ymax`, `width`, `height`). |
| **`faces[i].bounding_box_pixel`** | `Object` | Hộp bao khuôn mặt được chuyển đổi trực tiếp sang tọa độ pixel thực tế trên ảnh. |
| **`faces[i].blendshapes`** | `Array` | Điểm tin cậy (từ `0.0` đến `1.0`) cho 52 nhóm cơ mặt chuẩn hóa (vận động mí mắt, miệng, chân mày,...). |
| **`faces[i].facial_transformation_matrix`** | `Array` | Ma trận chuyển đổi 4x4 đại diện cho phép quay và tịnh tiến của khuôn mặt so với camera, thường dùng để dựng mô hình 3D hoặc xác định hướng nhìn. |
