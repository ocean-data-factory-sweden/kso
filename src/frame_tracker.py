import sys
import cv2
from random import randint

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


def createTrackerByName(trackerType):
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


def track_objects(video, class_ids, bboxes, start_frame, last_frame):

    # Set video to load
    colors = [(randint(0, 255)) for i in bboxes]

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Extract relevant frame
    # reader = Videos()
    # video = reader.read([videoPath])
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
