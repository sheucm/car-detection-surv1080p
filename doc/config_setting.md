# How to set config file

## Config zoo
You can download the model you like from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md),
and related sample config files is [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) provided by google researchers.

## Should-modified parameters
- fine_tune_checkpoint: path to model checkpoint.
- train_input_reader.tf_record_input_reader.input_path: path to train tf record file.
- train_input_reader.label_map_path: path to label map
- eval_input_reader.tf_record_input_reader.input_path: path to test tf record file.
- eval_input_reader.label_map_path: path to label map

