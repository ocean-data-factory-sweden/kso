import os, cv2


# Calculate length and fps of a movie
def get_length(video_file):
    
    #final_fn = video_file if os.path.isfile(video_file) else koster_utils.unswedify(video_file)
    
    if os.path.isfile(video_file):
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)     
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = frame_count/fps
    else:
        print("Length and fps for", video_file, "were not calculated")
        length, fps = None, None
        
    return fps, length
