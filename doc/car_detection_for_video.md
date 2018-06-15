# Run car detection for video

Make sure you have model config file, frozen inference graph file,
and opencv 3 installed. We'll use opencv3 to read video frame and
use tracker to stable the detection result.

Run following command ...
``` bash
python car_detection_for_video.py
```


Note: Remember to set python environment

``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
