import os


from data_loaders.data_utils import read_video_cv2, display_video_as_animation, reencode_video_ffmpeg, \
     run_darknet_on_video, run_darknet_on_video_batch, run_darknet_on_video_batch_with_line, stick_videos

if __name__=='__main__':
    # root_dir = 'C:/Users/Woo-Jin/workspace/blue_altex_counting_detection'
    root_dir = '/workspace'
    encoded_path = os.path.join(root_dir, 'data/encoded')

    encode_video = 0
    frame_skip=False
    resolution=(1920, 1080)
    if encode_video:
        ## encode video
        os.makedirs(encoded_path, exist_ok=True)
        input_mp4_path = os.path.join(root_dir, 'data/raw/TokiwaLance5_110123_C2_clipped.MP4')

        output_path = os.path.join(encoded_path, f'TokiwaLance5_110123_C2_clipped_{resolution[0]}_frameskip_{frame_skip}.MP4')
        reencode_video_ffmpeg(input_mp4_path, output_path, frame_skip=frame_skip, resolution=resolution)

    display_video = 0
    if display_video:
        mp4_path = os.path.join(encoded_path, f'TokiwaLance5_110123_C2_clipped_{resolution[0]}_frameskip_{frame_skip}.MP4')
        video_frames = read_video_cv2(os.path.join(encoded_path, os.path.basename(mp4_path)), n_frames=1e5)
        print('video_frames =', video_frames.shape)
        display_video_as_animation(video_frames)



    track_fish = 0
    from argparse import Namespace

    if track_fish:
        config_path = os.path.join(root_dir, 'saved_models/deep_yolo-fish-1/yolo-fish-2-merge.cfg')
        weights_path = os.path.join(root_dir, 'saved_models/deep_yolo-fish-1/merge_yolo-fish-2.weights')
        # encoded_video_path = 'C:/Users/Woo-Jin/workspace/blue_altex_counting_detection/data/encoded/TokiwaLance5_110123_C2_clipped.MP4'
        # encoded_video_path = os.path.join(encoded_path, f'TokiwaLance5_110123_C2_clipped_{resolution[0]}_frameskip_{frame_skip}.MP4')
        
        # input_video_path = os.path.join(root_dir, 'data/raw/TokiwaLance5_110123_C2_clipped.MP4')  # to_left
        input_video_path = os.path.join(root_dir, 'data/raw/Corral H.MP4')

        tracker_params = {'max_age': 5, 'n_init': 5, 'max_iou_dist': 0.7}
        output_folder = os.path.join(root_dir, 'output', os.path.basename(input_video_path)[:-4])
        config_params = Namespace(**{
                                    'config_path': config_path,
                                    'weights_path': weights_path,
                                    'tracker_params': tracker_params,
                                    'input_video_path': input_video_path,
                                    'output_folder': output_folder,
                                    'batch_size': 32,
                                    'main_direction': 'to_left',
                                    'save_interval': 384  # units of frames (12 batches)
                                    })
        run_darknet_on_video_batch_with_line(config_params)   
                

    merge_videos = 1
    if merge_videos:
        videos_folder = os.path.join(root_dir, 'output', 'Corral H_maxage_5_ninit_5_mid_0.7')
        stick_videos(videos_folder)



        # def run_darknet_and_display():
        # """
        # Run YOLO detection using Darknet executable from Python.
        # """

        # # Define the command as a list
        # command = [
        #     "darknet", "detect", 
        #     "cfg/yolov4.cfg", 
        #     "yolov4.weights", 
        #     "data/dog.jpg"
        # ]
        
        # # Define the working directory where 'darknet.exe' is located
        # working_directory = 'D:/darknet-master/darknet-master'

        # try:
        #     with open("darknet_output.log", "w") as logfile:
        #         process = subprocess.run(command, cwd=working_directory , stdout = logfile, shell=True)
        # except FileNotFoundError as e:
        #     print(f"Error: {e}. Ensure 'darknet.exe' exists in {working_directory}")
