import cv2
import os
import sys
import tempfile
import unittest
import pickle

# from importlib.resources import path as resource_path
# with resource_path("lib", "encode_feature.py") as encode_feature_path:
#     sys.path.append(os.path.dirname(str(encode_feature_path)))
#     from encode_feature import FeatureEncoder

sys.path.append(os.path.dirname(__file__) + "/..")
from lib.encode_feature import FeatureEncoder

class TestFeatureEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = FeatureEncoder()
        self.test_img_path = os.path.join(os.path.dirname(__file__), "sweetdog.jpeg")
        self.test_img = cv2.imread(self.test_img_path)
        self.test_kp = cv2.AKAZE_create().detect(self.test_img)
        

    def test_run(self):
        # テスト用の特徴量
        test_features = {
            "keypoint2d": self.test_kp,
            "descriptor2d": None,
        }
        self.encoder.run(self.test_img_path, test_features)
        self.assertTrue(self.test_img_path in self.encoder.database)
        self.assertTrue(isinstance(self.encoder.database[self.test_img_path], dict))
        self.assertTrue("keypoint2d" in self.encoder.database[self.test_img_path])
        self.assertTrue("descriptor2d" in self.encoder.database[self.test_img_path])
        self.assertEqual(len(self.encoder.database[self.test_img_path]["keypoint2d"]), len(self.test_kp))

    def test_save(self):
        test_features = {
            "keypoint2d": self.test_kp,
            "descriptor2d": None,
        }
        self.encoder.run(self.test_img_path, test_features)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "test.pickle")
            self.encoder.save(tmp_file_path)
            with open(tmp_file_path, mode="rb") as f:
                loaded_database = pickle.load(f)
            self.assertEqual(len(loaded_database), 1)
            self.assertTrue(self.test_img_path in loaded_database)
            self.assertTrue(isinstance(loaded_database[self.test_img_path], dict))
            self.assertTrue("keypoint2d" in loaded_database[self.test_img_path])
            self.assertTrue("descriptor2d" in loaded_database[self.test_img_path])
            self.assertEqual(len(loaded_database[self.test_img_path]["keypoint2d"]), len(self.test_kp))

if __name__ == '__main__':
    unittest.main()