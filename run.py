from lib.get_image_paths import *
from lib.feature_extractor import *
from lib.encode_feature import *
from lib.decode_feature import *

def encode_main():
    encoder = FeatureEncoder()
    image_paths = get_paths_image()
    for _, image_p in enumerate(image_paths):
        image = cv2.imread(str(image_p))
        feats = get_features(image)
        encoder.run(str(image_p), feats)
    encoder.save("database.pickle")

def decode_main():
    decoder = FeatureDecoder()
    feat_database:dict = decoder.run('database.pickle')
    feat_db_idx0 = list(feat_database.keys())[0]

    image = cv2.imread(feat_db_idx0)
    drawkpts = cv2.drawKeypoints(image, feat_database[feat_db_idx0]["keypoint2d"],
                                 outImage=None,
                                 color=(0,1,0),
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    raise

if __name__ == '__main__':
    # encode_main()
    decode_main()