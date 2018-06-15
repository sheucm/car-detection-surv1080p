# How to export inference graph


Run following command...
``` bash
python export_inference_graph.py
--input_type image_tensor \
--pipeline_config_path ssd_mobilenet_v1_coco.config \
--trained_checkpoint_prefix training/model.ckpt-001 \
--output_directory graph/ssd_mobilenet_v1 \
```


The `pipelien_config_path` is a model config file. <br>
The `trained_checkpoint_prefix` is checkpoint from `train_dir`
and change last three numbers. Check if checkpoint file `model.ckpt-XXXX`
has three files: `model.ckpt-XXXX.data-00000-of-00001`, `model.ckpt-XXXX.index`
and `model.ckpt-XXXX.meta`.


After executing, there is `frozen_inference_graph.pb` and some files in graph directory.


Note: Remember to set python environment

``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
