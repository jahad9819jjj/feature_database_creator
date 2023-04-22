import pickle

class FeatureEncoder:
    def __init__(self) -> None:
        self.database:dict = {}

    def run(self, path, features):
        keypoints = []
        for p in features["keypoint2d"]:
            keypoints.append(
            [
                p.pt,
                p.size,
                p.angle,
                p.response,
                p.octave,
                p.class_id
            ]
            )
            map(bytes, keypoints)
            self.database[path] = {
                "keypoint2d": keypoints,
                "descriptor2d": features["descriptor2d"]
            }
    
    def save(self, save_path)->bool:    
        with open(save_path, mode="wb") as file:
            success = pickle.dump(self.database, file)
        return success