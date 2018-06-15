# How to generate TF Record files

After converting xml to csv files,
or make sure there are data/train.csv and data/test.csv files,
execute following command.

```` bash
python generate_tfrecord.py
--csv_input=data/train_labels.csv
--output_path=data/train.record
--images_dir=images/train
````

```` bash
python generate_tfrecord.py
--csv_input=data/test_labels.csv
--output_path=data/test.record
--images_dir=images/test
````


Note: Remember to set python environment

``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```