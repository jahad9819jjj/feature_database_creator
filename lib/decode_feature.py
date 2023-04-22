import pickle
import cv2

class FeatureDecoder:
    def __init__(self) -> None:
        self.database:dict = {}
    
    def run(self, path)->dict:
        with open(path, mode='rb') as file:
            database = pickle.load(file)
        for path in database:
            items = database[path]
            list(map(list, items["keypoint2d"]))
            
            keypoints = []
            for keypoint in items["keypoint2d"]:
                keypoint = cv2.KeyPoint(
                    x=keypoint[0][0],
                    y=keypoint[0][1],
                    size=keypoint[1],
                    angle=keypoint[2],
                    response=keypoint[3],
                    octave=keypoint[4],
                    class_id=keypoint[5]
                )
                keypoints.append(keypoint)
            items["keypoint2d"] = keypoints
        return database