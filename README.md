# feature_database_creator

## What is this?
In feature detection, while maintaining the accuracy of the feature detector, it is inefficient to detect every database image in real time.

Therefore, the following program is designed to create a database of feature values.


## Usage
### Prepare dataset
Download the dataset for feature extraction in the following structure:

```:bash
├── data
│   ├── brandenburg_gate
│   │   ├── images
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   ├── british_museum
│   │   ├── calibration.csv
│   │   ├── images
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
```

### Run
To run the program, use the following command:

```
pip3 install -r requirements.txt
python3 run.py
```

Then you could watch the following output.
```
Encode[1] or Decode[2]?
```


`encode` is used to extract features from the input data and encode them as binary format.

`decode` is used to decode the binary-encoded features and check whether they can be reconstructed or not.