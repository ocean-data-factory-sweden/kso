import cv2

# from random import randint

trackerTypes = [
    "BOOSTING",
    "MIL",
    "KCF",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
    "CSRT",
]


def createTrackerByName(trackerType: str):
    """
    It creates a tracker based on the tracker name

    :param trackerType: The type of tracker we want to use
    :return: The tracker is being returned.
    """
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print("Incorrect tracker name")
        print("Available trackers are:")
        for t in trackerTypes:
            print(t)

    return tracker


def track_objects(video, class_ids: list, bboxes: list, start_frame: int, last_frame: int):
    """
    It takes a video, a list of bounding boxes, and a start and end frame, and returns a list of tuples
    containing the frame number, and the bounding box coordinates

    :param video: the video to be tracked
    :param class_ids: The class of the object you want to track
    :param bboxes: the bounding boxes of the objects to be tracked
    :param start_frame: the frame number to start tracking from
    :param last_frame: the last frame of the video to be processed
    :return: A list of tuples, where each tuple contains the frame number, x, y, width, and height of
    the bounding box.
    """

    # Set video to load
    # colors = [(randint(0, 255)) for i in bboxes]

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Extract relevant frame
    frame = video[start_frame]  # [0]

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    t_bbox = []
    t = 0
    # Process video and track objects
    for current_frame in range(start_frame + 1, last_frame + 1):
        frame = video[current_frame]  # [0]

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if success:
            t += 1
            for i, newbox in enumerate(boxes):
                t_bbox.append(
                    (t, int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3]))
                )

    return t_bbox
    # draw tracked objects
    #  for i, newbox in enumerate(boxes):
    #    p1 = (int(newbox[0]), int(newbox[1]))
    #    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    #    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    # cv2.imshow('MultiTracker', frame.astype('uint8') * 255)

    # quit on ESC button
    # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    #  break
