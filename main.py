import cv2
import numpy as np

# Detect "blips" in an image using Shi-Tomasi corner detection
def detect_blips(image, max_blips=100):
    feature_params = dict(maxCorners=max_blips,
                          qualityLevel=0.01,
                          minDistance=10,
                          blockSize=7)
    blips = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
    return np.int32(blips) if blips is not None else None

# Track blips across frames using Lucas-Kanade optical flow
def track_blips(prev_image, next_image, prev_blips):
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    
    # Ensure blips are of type float32 for optical flow calculation
    prev_blips = np.float32(prev_blips)
    
    # Calculate optical flow
    next_blips, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_blips, None, **lk_params)
    
    # Select good points based on the status array
    good_old = prev_blips[status == 1]
    good_new = next_blips[status == 1]
    
    return good_old, good_new

# Reconstruct 3D points from two frames using Essential Matrix
def reconstruct_3d_points(old_blips, new_blips, camera_matrix):
    # Estimate Essential matrix
    E, _ = cv2.findEssentialMat(new_blips, old_blips, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover the relative camera rotation and translation
    _, R, t, _ = cv2.recoverPose(E, new_blips, old_blips, camera_matrix)
    
    # Triangulate points to get the 3D structure
    proj_1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first camera
    proj_2 = np.hstack((R, t))                        # Projection matrix for the second camera
    
    # Triangulate to obtain 3D points in homogeneous coordinates
    points_4d_hom = cv2.triangulatePoints(proj_1, proj_2, old_blips.T, new_blips.T)
    
    # Convert from homogeneous coordinates to 3D
    points_3d = points_4d_hom / points_4d_hom[3]
    return points_3d[:3].T  # Return x, y, z coordinates

# Main function to process video and reconstruct 3D points
def process_video_for_3d_reconstruction(video_path, camera_matrix, max_blips=100):
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect blips in the first frame
    prev_blips = detect_blips(prev_gray, max_blips=max_blips)
    
    # Loop over the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track the blips from the previous frame to the current frame
        old_blips, new_blips = track_blips(prev_gray, next_gray, prev_blips)
        
        # Reconstruct 3D points
        points_3d = reconstruct_3d_points(old_blips, new_blips, camera_matrix)
        print("Reconstructed 3D points:", points_3d)
        
        # Visualize the tracking (optional)
        for i, (new, old) in enumerate(zip(new_blips, old_blips)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Convert coordinates to integers for OpenCV functions
            a, b = int(a), int(b)
            c, d = int(c), int(d)
            
            # Draw a circle around the new position of the blip
            cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
            
            # Draw a line connecting the previous and current positions
            cv2.line(frame, (a, b), (c, d), (0, 255, 255), 2)
        
        # Show the result
        cv2.imshow('Blip Tracking and 3D Reconstruction', frame)
        
        # Update previous frame and points for the next iteration
        prev_gray = next_gray.copy()
        prev_blips = new_blips.reshape(-1, 1, 2)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example camera matrix (for a 720p camera)
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]])

# Video path to process
video_path = 'flood.mp4'

# Run the 3D reconstruction pipeline
process_video_for_3d_reconstruction(video_path, camera_matrix)
