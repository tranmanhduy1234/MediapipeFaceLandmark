import argparse
import cv2
import json
import mediapipe as mp
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def main():
    parser = argparse.ArgumentParser(description="Detect face landmarks from a single image using MediaPipe Face Landmarker.")
    parser.add_argument("--image", type=str, default="image.jpg", help="Path to the input image.")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save the output JSON file.")
    parser.add_argument("--model", type=str, default="face_landmarker.task", help="Path to the face landmarker model file.")
    parser.add_argument("--num_faces", type=int, default=1, help="Maximum number of faces to detect.")
    parser.add_argument("--draw-output", type=str, default="output_drawn.jpg", help="Path to save the image with drawn landmarks. Set to empty string to skip drawing.")
    args = parser.parse_args()

    # Verify model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Please make sure 'face_landmarker.task' exists in the directory or specify its path using --model.")
        sys.exit(1)

    # Verify input image
    if not os.path.exists(args.image):
        print(f"Error: Input image file '{args.image}' not found.")
        sys.exit(1)

    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image '{args.image}'.")
        sys.exit(1)

    height, width, channels = image.shape
    print(f"Image dimensions: {width}x{height} ({channels} channels)")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Configure Face Landmarker options
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=args.num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )

    print("Initializing Face Landmarker...")
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Running detection...")
        result = landmarker.detect(mp_image)

    # Prepare data to save to JSON
    output_data = {
        "image_info": {
            "path": os.path.abspath(args.image),
            "width": width,
            "height": height
        },
        "faces": []
    }

    if not result.face_landmarks:
        print("No faces detected in the image.")
    else:
        num_detected = len(result.face_landmarks)
        print(f"Successfully detected {num_detected} face(s).")
        
        for face_idx, landmarks in enumerate(result.face_landmarks):
            face_data = {
                "face_index": face_idx,
                "landmarks_normalized": [],
                "landmarks_pixel": [],
                "blendshapes": [],
                "facial_transformation_matrix": None
            }

            # Extract normalized & pixel landmarks
            for lm_idx, landmark in enumerate(landmarks):
                # Normalized coordinates (range 0.0 to 1.0)
                face_data["landmarks_normalized"].append({
                    "index": lm_idx,
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z)
                })

                # Pixel coordinates (x, y mapped to image resolution)
                pixel_x = int(landmark.x * width)
                pixel_y = int(landmark.y * height)
                # z is scaled relative to depth, approximately in the same scale as x in pixels
                pixel_z = float(landmark.z * width)
                face_data["landmarks_pixel"].append({
                    "index": lm_idx,
                    "x": pixel_x,
                    "y": pixel_y,
                    "z": pixel_z
                })

            # Calculate bounding box
            x_norms = [float(lm.x) for lm in landmarks]
            y_norms = [float(lm.y) for lm in landmarks]
            xmin_norm, xmax_norm = min(x_norms), max(x_norms)
            ymin_norm, ymax_norm = min(y_norms), max(y_norms)

            face_data["bounding_box_normalized"] = {
                "xmin": xmin_norm,
                "ymin": ymin_norm,
                "xmax": xmax_norm,
                "ymax": ymax_norm,
                "width": xmax_norm - xmin_norm,
                "height": ymax_norm - ymin_norm
            }

            xmin_px = int(xmin_norm * width)
            ymin_px = int(ymin_norm * height)
            xmax_px = int(xmax_norm * width)
            ymax_px = int(ymax_norm * height)

            face_data["bounding_box_pixel"] = {
                "xmin": xmin_px,
                "ymin": ymin_px,
                "xmax": xmax_px,
                "ymax": ymax_px,
                "width": xmax_px - xmin_px,
                "height": ymax_px - ymin_px
            }

            # Extract blendshapes (e.g. eyeBlinkLeft, mouthSmileRight, etc.)
            if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                for blendshape in result.face_blendshapes[face_idx]:
                    face_data["blendshapes"].append({
                        "category_name": blendshape.category_name,
                        "score": float(blendshape.score)
                    })

            # Extract facial transformation matrix
            if result.facial_transformation_matrixes and face_idx < len(result.facial_transformation_matrixes):
                matrix = result.facial_transformation_matrixes[face_idx]
                face_data["facial_transformation_matrix"] = [list(map(float, row)) for row in matrix]

            output_data["faces"].append(face_data)

        # Draw landmarks if requested
        if args.draw_output:
            print(f"Drawing landmarks and saving to {args.draw_output}...")
            drawn_image = image.copy()
            
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh
            
            for face_idx, face_landmarks in enumerate(result.face_landmarks):
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks
                ])
                
                # Draw tesselation (face surface mesh)
                mp_drawing.draw_landmarks(
                    image=drawn_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw contours (eyes, eyebrows, mouth, face silhouette)
                mp_drawing.draw_landmarks(
                    image=drawn_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw irises
                mp_drawing.draw_landmarks(
                    image=drawn_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Draw bounding box
                xs = [lm.x * width for lm in face_landmarks]
                ys = [lm.y * height for lm in face_landmarks]
                x_min_px, x_max_px = int(min(xs)), int(max(xs))
                y_min_px, y_max_px = int(min(ys)), int(max(ys))
                
                cv2.rectangle(drawn_image, (x_min_px - 10, y_min_px - 10), (x_max_px + 10, y_max_px + 10), (0, 200, 255), 2)
                cv2.putText(drawn_image, f"Face #{face_idx}", (x_min_px - 10, max(y_min_px - 15, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                
            cv2.imwrite(args.draw_output, drawn_image)
            print("Visualized image saved.")

    # Save to JSON file
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()
