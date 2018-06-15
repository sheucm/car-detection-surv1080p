# How to train model

If you haven't set up, go [here](installation.md) for installation.

Run following command...
``` bash
python /PATH_TO/Object_Detection_API/models/research/object_detection/train.py \
--logtostderr \
--train_dir=training \
--pipeline_config_path=ssd_mobilenet_v1_coco.config
```

`train_dir` is where the checkpoints and summaries output.
You can use tensorboard to look up summaries form this directory.

``` bash
tensorboard --logdir=training
```


Note: Remember to set python environment

``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
