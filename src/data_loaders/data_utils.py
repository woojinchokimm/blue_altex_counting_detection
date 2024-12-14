import os
import subprocess
import traceback
import multiprocessing

import ffmpeg
import time
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deep_sort_realtime.deepsort_tracker import DeepSort


def display_video_as_animation(frames, interval=100):
    """
    Displays a video stored in a numpy array as an animation using matplotlib.

    Parameters:
    - frames: numpy array of shape (n_frames, height, width, 3) representing the video.
    - interval: Time in milliseconds between frames (default is 100ms).
    """
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Display the first frame as the initial image
    im = ax.imshow(frames[0])

    # Function to update the frame for the animation
    def update(frame):
        im.set_array(frames[frame])  # Update the displayed image
        ax.set_title(f"Frame: {frame}")  # Update the title with the frame number
        return [im]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)

    # Display the animation
    plt.show()
    # plt.close(fig)



def read_video_cv2(filename, n_frames=1e3):
    cap = cv2.VideoCapture(filename)
    all_frames = []
    i = 0
    
    # Get the total number of frames in the video if possible
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames =', total_frames)
    n_frames = min(n_frames, total_frames)  # Ensure we don't exceed the total number of frames
    
    # Use tqdm to wrap the loop for progress bar
    with tqdm(total=n_frames, desc="Reading frames", unit="frame") as pbar:
        while cap.isOpened() and i < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            np_frame = np.array(frame)
            all_frames.append(np_frame)
            i += 1
            pbar.update(1)  # Update the progress bar after processing each frame
    
    return np.array(all_frames)



def reencode_video_ffmpeg(input_path, output_path, frame_skip=3, resolution=(416, 416)):
    """
    Re-encodes a video to H.264 format, resizes frames, skips frames, and removes its audio using ffmpeg-python.

    Parameters:
    - input_path: Path to the input video file.
    - output_path: Path to the output video file.
    - frame_skip: Number of frames to skip (e.g., 10 means keep every 10th frame).
    - resolution: Target resolution as a tuple (width, height), default is (416, 416).
    """
    try:
        # Probe to get the original frame rate
        probe = ffmpeg.probe(input_path)
        original_fps = eval(next(
            stream['avg_frame_rate'] for stream in probe['streams'] if stream['codec_type'] == 'video'
        ))
        # Adjust the frame rate based on frame_skip
        if frame_skip and frame_skip > 1:
            new_fps = original_fps / frame_skip
        else:
            new_fps = original_fps  # No frame skipping, retain the original frame rate

        # Run FFmpeg with resizing and skipping frames
        (
            ffmpeg
            .input(input_path)  # Input video
            .output(
                output_path,
                # vf=f"scale={resolution[0]}:{resolution[1]}",  # Resize to 416x416
                r=new_fps,  # Adjust frame rate to skip frames
                vcodec='libx264',  # Use H.264 codec
                preset='fast',  # Encoding speed/quality balance
                crf=23,  # Quality control
                an=None  # Remove audio
            )
            .overwrite_output()  # Allow overwriting existing files
            .run()
        )
        print(f"Processed video saved as: {output_path}")
    except Exception as e:
        print(f"Error occurred during re-encoding: {e})")



def read_video_cv2_resilient(filename, max_retries=5):
    """
    Reads frames from a video file and handles cases where frames might be missing.
    
    Parameters:
    - filename: Path to the video file.
    - max_retries: Maximum number of retries for reading a single frame (default is 5).
    
    Returns:
    - numpy array of frames.
    """
    cap = cv2.VideoCapture(filename)

    # Get the total number of frames in the video
    expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []  # List to store frames
    frame_index = 0  # Frame index for tracking

    while frame_index < expected_frame_count:
        retries = 0  # Retry counter for the current frame

        while retries < max_retries:
            ret, frame = cap.read()

            if ret and frame is not None:
                # Successfully read the frame
                np_frame = np.array(frame)
                all_frames.append(np_frame)
                break  # Exit the retry loop and move to the next frame
            else:
                retries += 1
                print(f"Frame {frame_index} is empty. Retrying {retries}/{max_retries}...")

        if retries >= max_retries:
            print(f"Skipping frame {frame_index} after {max_retries} failed attempts.")
        
        frame_index += 1  # Increment the frame index regardless of success or failure

    cap.release()  # Release the video capture
    return np.array(all_frames)



def run_darknet_on_video(config_path, weights_path, video_path, output_folder, output_video_path=None):
    """
    Run YOLO detection using Darknet on an MP4 video and visualize/save the results.

    Parameters:
    - config_path: Path to YOLO configuration file.
    - weights_path: Path to YOLO weights file.
    - video_path: Path to the input video (MP4 format supported).
    - output_folder: Path to save the output video with predictions.
    - output_video_path: Path to save the processed video (optional).
    
    Returns:
    - None: Displays the processed video with bounding boxes.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_age=10,  # Maximum number of frames to retain an object after detection is lost
        n_init=3,    # Minimum detections required to confirm a track
        max_iou_distance=0.7  # Maximum IOU distance for matching detections
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer for saving output
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define the working directory where 'darknet.exe' is located
    working_directory = 'D:/darknet-master/darknet-master'


    try:
        frame_count = 0
        unique_fish_count = set()  # To keep track of unique fish IDs

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip the first 100 frames
            if frame_count < 250:
                frame_count += 1
                continue

            # Save the current frame to disk as an image
            frame_image_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_image_path, frame)

            # Define the command for Darknet detection
            command = [
                "darknet", "detect",
                config_path,
                weights_path,
                frame_image_path
            ]
 
            # Run Darknet detection
            try:
                with open("darknet_output.log", "w") as logfile:
                    process = subprocess.run(command, cwd=working_directory , stdout = logfile, shell=True)
            except FileNotFoundError as e:
                print(f"Error: {e}. Ensure 'darknet.exe' exists in {working_directory}")


            # Load the prediction image generated by Darknet
            prediction_image_path = os.path.join(working_directory, "predictions.jpg")
            if os.path.exists(prediction_image_path):
                processed_frame = cv2.imread(prediction_image_path)
                if processed_frame is not None:
                    # Write the processed frame to the output video
                    if out:
                        out.write(processed_frame)

                    # Display the processed frame
                    cv2.imshow("YOLO Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f"Warning: Could not read processed frame for frame {frame_count}")
            else:
                print(f"Warning: No prediction image generated for frame {frame_count}")

            frame_count += 1

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


def run_darknet_on_video(config_path, weights_path, names_path, video_path, output_folder, output_video_path=None):
    """
    Run YOLO detection using OpenCV's DNN module and track objects using DeepSort.

    Parameters:
    - config_path: Path to YOLO configuration file.
    - weights_path: Path to YOLO weights file.
    - names_path: Path to class names file.
    - video_path: Path to the input video (MP4 format supported).
    - output_folder: Path to save the output video with predictions.
    - output_video_path: Path to save the processed video (optional).
    
    Returns:
    - None: Displays the processed video with bounding boxes and counts tracked objects.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model with OpenCV
    print(cv2.cuda.getCudaEnabledDeviceCount())
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available

    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_age=30,  # Maximum number of frames to retain an object after detection is lost
        n_init=3,    # Minimum detections required to confirm a track
        max_iou_distance=0.7  # Maximum IOU distance for matching detections
    )

    # Original video dimensions and YOLO input dimensions
    original_width, original_height = 1920, 1080
    yolo_input_width, yolo_input_height = 608, 608

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    # Video writer for saving output
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize counters
    frame_count = 0
    unique_fish_count = set()  # To keep track of unique fish IDs

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(yolo_input_width, yolo_input_height), swapRB=True, crop=False)
            net.setInput(blob)

            # Perform forward pass
            layer_names = net.getUnconnectedOutLayersNames()
            start = time.time()
            detections = net.forward(layer_names)
            end = time.time()
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

            detections_for_tracking = []
            boxes = []
            confidences = []

            for output in detections:  # Iterate through each YOLO detection layer
                for detection in output:
                    confidence = detection[4]  # Confidence of the bounding box

                    if confidence > 0.5:  # Apply a confidence threshold to filter weak detections
                        # Extract and scale bounding box coordinates
                        center_x = int(detection[0] * width)  # Scale center_x to the image width
                        center_y = int(detection[1] * height)  # Scale center_y to the image height
                        box_width = int(detection[2] * width)  # Scale width to the image width
                        box_height = int(detection[3] * height)  # Scale height to the image height

                        # Calculate top-left corner of the bounding box
                        x_min = int(center_x - box_width / 2)
                        y_min = int(center_y - box_height / 2)

                        # Store the bounding box and confidence
                        boxes.append([x_min, y_min, box_width, box_height])
                        confidences.append(float(confidence))

            # Apply Non-Maximum Suppression (NMS)
            # indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

            # Filter boxes that survived NMS
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, box_width, box_height = boxes[i]
                    confidence = confidences[i]

                    # Append the surviving detections in the format required by DeepSort
                    detections_for_tracking.append(([x_min, y_min, box_width, box_height], confidence, "fish"))  # "fish" for single class

            # Perform tracking with DeepSort
            tracks = tracker.update_tracks(detections_for_tracking, frame=frame)

            # Draw tracks and update the unique fish count
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()  # Bounding box format: Left-Top-Right-Bottom
                x_min, y_min, x_max, y_max = map(int, bbox)  # Ensure integer coordinates
                unique_fish_count.add(track_id)  # Add to unique fish count
                # Draw the bounding box and ID on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Overlay text: Number of fish detected and frame information
            cv2.putText(frame, f"Fish Detected: {len(unique_fish_count)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write the processed frame to the output video
            if out:
                out.write(frame)

            # Display the processed frame
            cv2.imshow("YOLO Detection + Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()  #

    finally:
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print(f"Total unique objects detected: {len(unique_fish_count)}")


from torchvision.ops import batched_nms
import torch

def run_darknet_on_video_batch(config_path, weights_path, video_path, output_folder, batch_size=32):
    """
    Run YOLO detection using OpenCV's DNN module and track objects using DeepSort with batch inference.

    Parameters:
    - config_path: Path to YOLO configuration file.
    - weights_path: Path to YOLO weights file.
    - video_path: Path to the input video (MP4 format supported).
    - output_folder: Path to save the output video with predictions.
    - batch_size: Number of frames to process in a single batch.
    
    Returns:
    - None: Displays the processed video with bounding boxes and counts tracked objects.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    import multiprocessing
    cv2.setNumThreads(multiprocessing.cpu_count())

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available

    # Initialize DeepSort tracker
    max_age = 3  # specifies after how many frames unallocated tracks will be deleted.
    n_init = 5 # specifies after how many frames newly allocated tracks will be activated.
    max_iou_dist = 0.7 # threshold value that determines how much the bounding boxes should overlap to determine the identity of the unassigned track.

    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_dist
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_input_size = (608, 608)

    # Video writer for saving output
    out = None
    if output_folder:
        save_file = os.path.join(output_folder, f'results_maxage_{max_age}_ninit_{n_init}_mid_{max_iou_dist}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_file, fourcc, fps, (width, height))

    # Initialize counters and batching
    frame_count = 0
    unique_fish_count = set()
    batch_frames = []
    total_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            batch_frames.append(frame)

            # if frame_count > 1000:
            #     break

            start = time.time()
            # Process batch when full or at the end of the video
            if len(batch_frames) == batch_size or frame_count == total_frames - 1:
                # Prepare blob for the batch
                blob = cv2.dnn.blobFromImages(batch_frames, scalefactor=1/255.0, size=model_input_size, swapRB=True, crop=False)
                net.setInput(blob)

                # Perform forward pass
                layer_names = net.getUnconnectedOutLayersNames()
                outputs = net.forward(layer_names)

                all_boxes = []
                all_scores = []
                all_idxs = []

                # Gather detections for all frames in the batch
                for batch_idx, frame in enumerate(batch_frames):
                    frame_boxes = []
                    frame_scores = []

                    # Gather detections across all YOLO layers for this frame
                    for layer_idx, output in enumerate(outputs):
                        detections = output[batch_idx]  # Detections for the current frame
                        for detection in detections:
                            confidence = float(detection[4])  # Extract objectness confidence
                            if confidence > 0.5:  # Confidence threshold
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                box_width = int(detection[2] * width)
                                box_height = int(detection[3] * height)
                                x_min = int(center_x - box_width / 2)
                                y_min = int(center_y - box_height / 2)

                                # Store bounding box and confidence
                                frame_boxes.append([x_min, y_min, x_min + box_width, y_min + box_height])  # (x1, y1, x2, y2)
                                frame_scores.append(confidence)

                    # Add detections for this frame to the global list
                    all_boxes.extend(frame_boxes)
                    all_scores.extend(frame_scores)
                    all_idxs.extend([batch_idx] * len(frame_boxes))  # Assign all boxes for this frame to the same idx

                # Convert to tensors
                all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
                all_scores = torch.tensor(all_scores, dtype=torch.float32)
                all_idxs = torch.tensor(all_idxs, dtype=torch.int64)

                # Apply batched NMS
                iou_threshold = 0.4
                indices = batched_nms(boxes=all_boxes, scores=all_scores, idxs=all_idxs, iou_threshold=iou_threshold)

                # Filter selected boxes, scores, and idxs
                selected_boxes = all_boxes[indices]
                selected_scores = all_scores[indices]
                selected_idxs = all_idxs[indices]

                # List to store processed frames
                processed_frames = []
                
                # Split results back to frames
                for batch_idx in range(len(batch_frames)):
                    frame_indices = (selected_idxs == batch_idx).nonzero(as_tuple=True)[0]
                    frame_boxes = selected_boxes[frame_indices]
                    frame_scores = selected_scores[frame_indices]

                    # Prepare detections for DeepSort
                    detections_for_tracking = []
                    for box, score in zip(frame_boxes, frame_scores):
                        x_min, y_min, x_max, y_max = map(int, box)
                        detections_for_tracking.append(([x_min, y_min, x_max - x_min, y_max - y_min], score))  # (ltrb, score)

                    # Convert to DeepSort input format
                    detections_for_tracking = [
                        (bbox, confidence, "fish") for bbox, confidence in detections_for_tracking
                    ]  # Use "fish" as the label for single class

                    # Update the DeepSort tracker
                    tracks = tracker.update_tracks(detections_for_tracking, frame=batch_frames[batch_idx])

                    # Draw bounding boxes and track IDs
                    for track in tracks:
                        if not track.is_confirmed():
                            continue  # Skip unconfirmed tracks
                        track_id = track.track_id
                        unique_fish_count.add(track_id)  # Add track ID to the unique fish count set

                        # Draw bounding box and track ID
                        bbox = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(batch_frames[batch_idx], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(batch_frames[batch_idx], f"ID: {track_id}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Add overlay text for unique fish count and frame info
                    cv2.putText(batch_frames[batch_idx], f"Unique Fish Detected: {len(unique_fish_count)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(batch_frames[batch_idx], f"Frame: {frame_count + batch_idx + 1}/{total_frames}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Add frame to processed frames
                    processed_frames.append(batch_frames[batch_idx])

                    # # Optionally: Draw the selected bounding boxes on the frame
                    # for box, score in zip(frame_boxes, frame_scores):
                    #     x_min, y_min, x_max, y_max = map(int, box)
                    #     cv2.rectangle(batch_frames[batch_idx], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    #     cv2.putText(batch_frames[batch_idx], f"Conf: {score:.2f}", (x_min, y_min - 10),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # # Add overlay text for unique fish count and frame info
                    # cv2.putText(batch_frames[batch_idx], f"Unique Fish Detected: {len(unique_fish_count)}", (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # cv2.putText(batch_frames[batch_idx], f"Frame: {frame_count + batch_idx + 1}/{total_frames}", (10, 60),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # # Add frame to processed frames
                    # processed_frames.append(batch_frames[batch_idx])

                # Sequentially display frames
                while processed_frames:
                    frame = processed_frames.pop(0)  # Get the first frame from the list
                    cv2.imshow("YOLO Detection + Tracking", frame)
                    if out:
                        out.write(frame)

                    # Display the frame and check for exit
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

                frame_count += len(batch_frames)

                # Clear batch
                batch_frames = []

                end = time.time()
                batch_time = end - start
                remaining_frames = total_frames - frame_count
                remaining_batches = (remaining_frames + batch_size - 1) // batch_size
                eta_completion_time = remaining_batches * batch_time
                total_time += batch_time
                print(f"[INFO] Detector took {batch_time:.2f} seconds for a batch size of {batch_size}")
                print(f"[INFO] ETA to completion: {eta_completion_time / 60:.2f} minutes.")
                print(f"[INFO] Total time elapsed: {total_time / 60:.2f} minutes.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

    finally:
        cap.release()
        if out:
            print('Wait 10 seconds to finish processing for release..')
            cv2.waitKey(10000) 
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print(f"Total unique objects detected: {len(unique_fish_count)}")




def run_darknet_on_video_batch_with_line2(config_path, weights_path, video_path, output_folder, batch_size=32):
    """
    Run YOLO detection using OpenCV's DNN module and track objects using DeepSort with batch inference.

    Parameters:
    - config_path: Path to YOLO configuration file.
    - weights_path: Path to YOLO weights file.
    - video_path: Path to the input video (MP4 format supported).
    - output_folder: Path to save the output video with predictions.
    - batch_size: Number of frames to process in a single batch.

    Returns:
    - None: Displays the processed video with bounding boxes and counts tracked objects.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    import multiprocessing
    cv2.setNumThreads(multiprocessing.cpu_count())

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available

    # Initialize DeepSort tracker
    max_age = 3
    n_init = 5
    max_iou_dist = 0.7

    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_dist
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_input_size = (608, 608)

    # Define boundary line
    BOUNDARY_LINE_X = width // 2
    print('BOUNDARY_LINE_X =', BOUNDARY_LINE_X)

    # Counters for fish crossing the boundary
    left_to_right_count = 0
    right_to_left_count = 0

    # Dictionary to track the last known zone of each fish
    fish_last_zone = {}

    # Video writer for saving output
    out = None
    if output_folder:
        save_file = os.path.join(output_folder, f'results_maxage_{max_age}_ninit_{n_init}_mid_{max_iou_dist}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_file, fourcc, fps, (width, height))

    frame_count = 0
    batch_frames = []
    total_time = 0

    try:
        while True:
            ret, frame = cap.read()
            # if not ret:
            #     continue
            if not ret:
                if frame_count < total_frames:
                    continue
                else:
                    break

            batch_frames.append(frame)

            start = time.time()
            if len(batch_frames) == batch_size or frame_count == total_frames - 1:
                blob = cv2.dnn.blobFromImages(batch_frames, scalefactor=1/255.0, size=model_input_size, swapRB=True, crop=False)
                net.setInput(blob)

                layer_names = net.getUnconnectedOutLayersNames()
                outputs = net.forward(layer_names)

                all_boxes = []
                all_scores = []
                all_idxs = []

                for batch_idx, frame in enumerate(batch_frames):
                    frame_boxes = []
                    frame_scores = []

                    for layer_idx, output in enumerate(outputs):
                        detections = output[batch_idx]
                        for detection in detections:
                            confidence = float(detection[4])
                            if confidence > 0.5:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                box_width = int(detection[2] * width)
                                box_height = int(detection[3] * height)
                                x_min = int(center_x - box_width / 2)
                                y_min = int(center_y - box_height / 2)

                                frame_boxes.append([x_min, y_min, x_min + box_width, y_min + box_height])
                                frame_scores.append(confidence)

                    all_boxes.extend(frame_boxes)
                    all_scores.extend(frame_scores)
                    all_idxs.extend([batch_idx] * len(frame_boxes))

                all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
                all_scores = torch.tensor(all_scores, dtype=torch.float32)
                all_idxs = torch.tensor(all_idxs, dtype=torch.int64)

                iou_threshold = 0.4
                indices = batched_nms(boxes=all_boxes, scores=all_scores, idxs=all_idxs, iou_threshold=iou_threshold)

                selected_boxes = all_boxes[indices]
                selected_scores = all_scores[indices]
                selected_idxs = all_idxs[indices]

                processed_frames = []

                for batch_idx in range(len(batch_frames)):
                    frame_indices = (selected_idxs == batch_idx).nonzero(as_tuple=True)[0]
                    frame_boxes = selected_boxes[frame_indices]
                    frame_scores = selected_scores[frame_indices]

                    detections_for_tracking = []
                    for box, score in zip(frame_boxes, frame_scores):
                        x_min, y_min, x_max, y_max = map(int, box)
                        detections_for_tracking.append(([x_min, y_min, x_max - x_min, y_max - y_min], score))

                    detections_for_tracking = [
                        (bbox, confidence, "fish") for bbox, confidence in detections_for_tracking
                    ]
                    print('detections_for_tracking =', detections_for_tracking)

                    tracks = tracker.update_tracks(detections_for_tracking, frame=batch_frames[batch_idx])

                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        bbox = track.to_ltrb()
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        center_x = (x_min + x_max) // 2

                        # Determine current zone
                        current_zone = "left" if center_x < BOUNDARY_LINE_X else "right"

                        # Check for crossing
                        if track_id in fish_last_zone:
                            previous_zone = fish_last_zone[track_id]
                            if previous_zone == "left" and current_zone == "right":
                                left_to_right_count += 1
                            elif previous_zone == "right" and current_zone == "left":
                                right_to_left_count += 1

                        fish_last_zone[track_id] = current_zone

                        # Draw bounding box and track ID
                        cv2.rectangle(batch_frames[batch_idx], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(batch_frames[batch_idx], f"ID: {track_id}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    net_fish_count = left_to_right_count - right_to_left_count

                    # Add overlay text for frame info
                    cv2.putText(batch_frames[batch_idx], f"Frame: {frame_count + batch_idx + 1}/{total_frames}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Draw the boundary line
                    cv2.line(batch_frames[batch_idx], (BOUNDARY_LINE_X, 0), (BOUNDARY_LINE_X, height), (0, 0, 255), 2)

                    cv2.putText(batch_frames[batch_idx], f"Net Fish Count: {net_fish_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(batch_frames[batch_idx], f"Left to Right: {left_to_right_count}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(batch_frames[batch_idx], f"Right to Left: {right_to_left_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    processed_frames.append(batch_frames[batch_idx])

                for frame in processed_frames:
                    cv2.imshow("YOLO Detection + Tracking", frame)
                    if out:
                        out.write(frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

                frame_count += len(batch_frames)
                batch_frames = []

                end = time.time()
                batch_time = end - start
                remaining_frames = total_frames - frame_count
                remaining_batches = (remaining_frames + batch_size - 1) // batch_size
                eta_completion_time = remaining_batches * batch_time
                total_time += batch_time
                print(f"[INFO] Detector took {batch_time:.2f} seconds for a batch size of {batch_size}")
                print(f"[INFO] ETA to completion: {eta_completion_time / 60:.2f} minutes.")
                print(f"[INFO] Total time elapsed: {total_time / 60:.2f} minutes.")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    print(f"Left to Right Count: {left_to_right_count}")
    print(f"Right to Left Count: {right_to_left_count}")
    print(f"Net Fish Count: {left_to_right_count - right_to_left_count}")






def process_batch(batch_frames, net, tracker, frame_count, model_input_size, width, height,
                  count_dict, fish_positions, fish_counted, fish_speeds, BOUNDARY_LINE_X, main_direction, total_frames):
    
    blob = cv2.dnn.blobFromImages(batch_frames, scalefactor=1/255.0, size=model_input_size, swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    all_boxes = []
    all_scores = []
    all_idxs = []

    for batch_idx, frame in enumerate(batch_frames):
        frame_boxes = []
        frame_scores = []

        for layer_idx, output in enumerate(outputs):
            detections = output[batch_idx]
            for detection in detections:
                confidence = float(detection[4])
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    box_width = int(detection[2] * width)
                    box_height = int(detection[3] * height)
                    x_min = int(center_x - box_width / 2)
                    y_min = int(center_y - box_height / 2)

                    frame_boxes.append([x_min, y_min, x_min + box_width, y_min + box_height])
                    frame_scores.append(confidence)

        all_boxes.extend(frame_boxes)
        all_scores.extend(frame_scores)
        all_idxs.extend([batch_idx] * len(frame_boxes))

    all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    all_idxs = torch.tensor(all_idxs, dtype=torch.int64)

    iou_threshold = 0.4
    indices = batched_nms(boxes=all_boxes, scores=all_scores, idxs=all_idxs, iou_threshold=iou_threshold)

    selected_boxes = all_boxes[indices]
    selected_scores = all_scores[indices]
    selected_idxs = all_idxs[indices]

    processed_frames = []
    desired_track_length = 3
    speed_threshold = 3

    for batch_idx in range(len(batch_frames)):
        frame_indices = (selected_idxs == batch_idx).nonzero(as_tuple=True)[0]
        frame_boxes = selected_boxes[frame_indices]
        frame_scores = selected_scores[frame_indices]

        detections_for_tracking = []
        for box, score in zip(frame_boxes, frame_scores):
            x_min, y_min, x_max, y_max = map(int, box)
            detections_for_tracking.append(([x_min, y_min, x_max - x_min, y_max - y_min], score))

        detections_for_tracking = [
            (bbox, confidence, "fish") for bbox, confidence in detections_for_tracking
        ]

        tracks = tracker.update_tracks(detections_for_tracking, frame=batch_frames[batch_idx])

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x_min, y_min, x_max, y_max = map(int, bbox)
            center_x = (x_min + x_max) // 2

            # Update the x-coordinate history for the fish
            if track_id not in fish_positions:
                fish_positions[track_id] = []
                fish_counted[track_id] = {"left_to_right": False, "right_to_left": False, "display_text": None, "text_timer": 0}
            else:
                # Calculate speed as the difference in x positions
                fish_speeds[track_id] = abs(center_x - fish_positions[track_id][-1])
            fish_positions[track_id].append(center_x)

            
            # Handle main_direction "to_left"
            if main_direction == "to_left":
                # For fish moving in the main direction (right to left), count based on speed
                if (
                    len(fish_positions[track_id]) > desired_track_length and
                    all(fish_positions[track_id][i] > fish_positions[track_id][i + 1] for i in range(len(fish_positions[track_id]) - 1)) and
                    not fish_counted[track_id]["right_to_left"] and
                    fish_speeds[track_id] > speed_threshold
                ):
                    count_dict['right_to_left_count'] += 1
                    fish_counted[track_id]["right_to_left"] = True
                    fish_counted[track_id]["display_text"] = "+1"
                    fish_counted[track_id]["text_timer"] = 100  # Display for 10 frames
                    # cv2.putText(frame, "+1", (center_x, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # For fish moving in the opposite direction, require boundary crossing
                elif (
                    any(pos < BOUNDARY_LINE_X for pos in fish_positions[track_id][-desired_track_length:]) and
                    fish_positions[track_id][-1] > BOUNDARY_LINE_X and
                    not fish_counted[track_id]["left_to_right"]
                ):
                    count_dict['left_to_right_count'] += 1
                    fish_counted[track_id]["left_to_right"] = True
                    # cv2.putText(frame, "-1", (center_x, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    fish_counted[track_id]["display_text"] = "-1"
                    fish_counted[track_id]["text_timer"] = 100  # Display for 10 frames

            # Handle main_direction "to_right"
            elif main_direction == "to_right":
                # For fish moving in the main direction (left to right), count based on speed
                if (
                    len(fish_positions[track_id]) > desired_track_length and
                    all(fish_positions[track_id][i] < fish_positions[track_id][i + 1] for i in range(len(fish_positions[track_id]) - 1)) and
                    not fish_counted[track_id]["left_to_right"] and
                    fish_speeds[track_id] > speed_threshold
                ):
                    count_dict['left_to_right_count'] += 1
                    fish_counted[track_id]["left_to_right"] = True
                    fish_counted[track_id]["display_text"] = "+1"
                    fish_counted[track_id]["text_timer"] = 100  # Display for 10 frames
                    # cv2.putText(frame, "+1", (center_x, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # For fish moving in the opposite direction, require boundary crossing
                elif (
                    any(pos > BOUNDARY_LINE_X for pos in fish_positions[track_id][-desired_track_length:]) and
                    fish_positions[track_id][-1] < BOUNDARY_LINE_X and
                    not fish_counted[track_id]["right_to_left"]
                ):
                    count_dict['right_to_left_count'] += 1
                    fish_counted[track_id]["right_to_left"] = True
                    fish_counted[track_id]["display_text"] = "-1"
                    fish_counted[track_id]["text_timer"] = 100  # Display for 10 frames
                    # cv2.putText(frame, "-1", (center_x, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            # Handle text display for each track +1 or -1 for ease of visual counting
            if fish_counted[track_id]["text_timer"] > 0:
                cv2.putText(batch_frames[batch_idx], fish_counted[track_id]["display_text"], (x_max, y_min - 10),  # Position relative to bounding box
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if fish_counted[track_id]["display_text"] == "+1" else (0, 0, 255), 3)
                fish_counted[track_id]["text_timer"] -= 1  # Decrement timer

            # Draw bounding box and track ID
            cv2.rectangle(batch_frames[batch_idx], (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(batch_frames[batch_idx], f"ID: {track_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # Display counters on the frame
        if main_direction == 'to_left':
            net_fish_count = count_dict['right_to_left_count'] - count_dict['left_to_right_count']
            left_to_right_color = (255, 0, 255)
            right_to_left_color = (0, 255, 0)
        else:
            net_fish_count = count_dict['left_to_right_count'] - count_dict['right_to_left_count']
            left_to_right_color = (0, 255, 0)
            right_to_left_color = (255, 0, 255)
            

        # Draw the boundary line
        cv2.line(batch_frames[batch_idx], (BOUNDARY_LINE_X, 0), (BOUNDARY_LINE_X, height), (0, 0, 255), 2)

        cv2.putText(batch_frames[batch_idx], f"Net Fish Count: {net_fish_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(batch_frames[batch_idx], f"---> : {count_dict['left_to_right_count']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, left_to_right_color, 2)
        cv2.putText(batch_frames[batch_idx], f"<--- : {count_dict['right_to_left_count']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, right_to_left_color, 2)
        
        # Add overlay text for frame info
        cv2.putText(batch_frames[batch_idx], f"Frame: {frame_count + batch_idx + 1}/{total_frames}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                

        processed_frames.append(batch_frames[batch_idx])


    return processed_frames




def run_darknet_on_video_batch_with_line(config):
    """
    Run YOLO detection using OpenCV's DNN module and track objects using DeepSort with batch inference.

    Parameters:
    - config: Namespace

    Returns:
    - None: Displays the processed video with bounding boxes and counts tracked objects.
    """
    config_path = config.config_path
    weights_path = config.weights_path
    tracker_params = config.tracker_params
    video_path = config.input_video_path
    output_folder = config.output_folder
    main_direction = config.main_direction
    batch_size = config.batch_size
    save_interval = config.save_interval


    assert main_direction in ['to_left', 'to_right']
    assert save_interval % batch_size == 0, f'Save_interval must be a multiple of batch_size.'

    # Load YOLO model
    cv2.setNumThreads(multiprocessing.cpu_count())

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available

    # Initialize DeepSort tracker
    max_age = tracker_params['max_age']
    n_init = tracker_params['n_init']
    max_iou_dist = tracker_params['max_iou_dist']

    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_dist
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_input_size = (608, 608)

    # Define boundary line
    BOUNDARY_LINE_X = width // 2

    # Counters for fish crossing the boundary
    count_dict = {"left_to_right_count": 0, "right_to_left_count": 0}

    # Dictionary to track the last known zone of each fish
    fish_positions = {}
    fish_counted = {}  # Track counting status for each fish
    fish_speeds = {}

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_folder = f'{output_folder}_maxage_{max_age}_ninit_{n_init}_mid_{max_iou_dist}'
    os.makedirs(save_folder, exist_ok=True)
    print(f'Settings output folder as {save_folder}..')

    frame_count = 0
    batch_frames = []
    total_time = 0
    saving_frames = []

    desired_width = 960
    desired_height = 540
    cv2.namedWindow("YOLO Detection + Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection + Tracking", desired_width, desired_height) 

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_count < total_frames:
                    continue
                else:
                    break
            
            # frame_count += 1
            # if frame_count < 24582-68:
            #     continue
            # if frame_count < 15000:
                # continue

            batch_frames.append(frame)

            start = time.time()

            
            if len(batch_frames) == batch_size or frame_count == total_frames - 1:
                processed_frames = process_batch(
                                            batch_frames, net, tracker, frame_count, 
                                            model_input_size, width, height,
                                            count_dict, 
                                            fish_positions, fish_counted, fish_speeds, 
                                            BOUNDARY_LINE_X, 
                                            main_direction, total_frames)
                
                # Sequentially display frames
                for frame in processed_frames:  # Iterate over the list without popping
                    cv2.imshow("YOLO Detection + Tracking", frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):  # Display the frame and check for exit
                        break
                
                frame_count += len(batch_frames)
                batch_frames = []

                end = time.time()
                batch_time = end - start
                remaining_frames = total_frames - frame_count
                remaining_batches = (remaining_frames + batch_size - 1) // batch_size
                eta_completion_time = remaining_batches * batch_time
                total_time += batch_time
                print(f"[INFO] Detector took {batch_time:.2f} seconds for a batch size of {batch_size}")
                print(f"[INFO] Current frame count is {frame_count} out of {total_frames} total frames ({frame_count / total_frames * 100:.2f}%)")
                print(f"[INFO] ETA to completion: {eta_completion_time / 60:.2f} minutes or {eta_completion_time / 3600:.2f} hours.")
                print(f"[INFO] Total time elapsed: {total_time / 60:.2f} minutes.")


                saving_frames.extend(processed_frames)
                # Save intermediate results at intervals
                if frame_count % save_interval == 0:
                    print(f'Saving part {frame_count // save_interval}..')
                    temp_save_file = os.path.join(save_folder, f'part_{frame_count // save_interval}.MP4')
                    temp_out = cv2.VideoWriter(temp_save_file, fourcc, fps, (desired_width, desired_height))
                    for processed_frame in saving_frames:
                        processed_frame = cv2.resize(processed_frame, (desired_width, desired_height))
                        temp_out.write(processed_frame)
                    temp_out.release()
                    saving_frames = []  # Clear the saved frames to save memory


    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        cap.release()
        if processed_frames:  # Check if there are any unsaved frames
            temp_save_file = os.path.join(save_folder, f'part_final.mp4')
            temp_out = cv2.VideoWriter(temp_save_file, fourcc, fps, (width, height))
            for processed_frame in processed_frames:
                temp_out.write(processed_frame)
            temp_out.release()
        cv2.destroyAllWindows()




import subprocess

def stick_videos(video_folder):
    """
    Joins multiple video parts into a seamless whole-video.
    
    Args:
        video_folder (str): Path to the folder containing video parts.
    """
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder {video_folder} does not exist.")

    # List and sort video files
    files = sorted(
        [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
    )

    if not files:
        raise ValueError(f"No MP4 video files found in folder: {video_folder}")

    # Create a file list for FFmpeg
    file_list_path = os.path.join(video_folder, "file_list.txt")
    with open(file_list_path, "w") as file_list:
        for file in files:
            file_list.write(f"file '{file}'\n")

    # Output file
    output_file = os.path.join(video_folder, "merged_video.mp4")

    # Run FFmpeg to concatenate videos
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Merged video saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")