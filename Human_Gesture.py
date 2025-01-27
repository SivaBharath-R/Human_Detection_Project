
import cv2
import mediapipe as mp
import time


# Initialize Mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Correct video path
video_path = r"C:\Users\rapar\OneDrive\Desktop\project\Project_2\video\video1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video at path: {video_path}")
    exit()

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("End of video or cannot read the video.")
        break

    # Resize the frame for display purposes
    height, width, _ = img.shape
    max_width = 1000  
    max_height = 800 
    scale = min(max_width / width, max_height / height)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Process the frame for pose detection
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrected color conversion
    result = pose.process(imgRGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(
            img, result.pose_landmarks, mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    # FPS calculation and display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
