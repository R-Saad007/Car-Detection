## This script detects cars and tracks whether they are within a certain region.
## Need to clone ByteTrack and Yolov5 repositories
### ByteTrack
```git clone https://github.com/ifzhang/ByteTrack.git```

```pip install -r requirements.txt```

```python3 setup.py develop```

```pip install cython_bbox```
### YOLOv5
```git clone https://github.com/ultralytics/yolov5.git```

```pip install -r requirements.txt```
## Directory structure:
### ByteTrack -> YoloV5 + handler.py + video_folder(videos)
### Script Execution
```python handler.py -vid_path (video path from script's directory)```
### Updates Needed (depends on the resolution of the video used)
designated_region_start = (xmin, ymin) (Enter the starting coordinates of your region here)  
designated_region_end = (xmax, ymax) (Enter the ending coordinates of your region here)
