# Hand Gesture Recognition.

If you are looking for a quick hand gesture recognition model, you are at the right place.

In the repository I have created two models, a classic machine learning model and a neural network model. which would help you to get started with recognition of hand gestures within no time.

| id |      model      | average accuracy | file |
|----------|:-------------:|:---:|:-------:|
| 1 |  K Nearest Neighbour | 95.1%  | [KnnModel.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/KnnModel.py)
| 2 |  Neural Network Model  | 97.8%|   [neuralModel.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/KnnModel.py)

## Run Recognition

- First of all I recommend using a virtual env
- Install all requirements
- There is a pretrained model already with a few symbols. this is for a quick run and to test the code. i would staunchly recommend to [create new dataset](#train-your-data) and train with it.
```
pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python neuralModel.py
```

## <a name="train-your-data"></a> Train With Your data.
before getting into creating a dataset and training, we have to understand the [config.json](https://github.com/gitmax681/hand-gesture-recognition/blob/master/config.json) file

```json
{
  "CurrentIndex": 5,
  "labels": [
    "swag",
    "peace",
    "call-me-hand",
    "thumbs-up",
    "heart"
  ],
  "DataVersion": 1.0,
  "ModelVersion": 1.0,
  "MaxSamples": 150,
  "CurrentModel": "model-v1.0.h5",
  "CurrentData": "data-v1.0.csv",
}
```
this is a very crucial file, a small mistake in this would mess up whole model's predictions

- CurrentIndex - used to determine the index while generating data
- labels - used to get the acutal word from a predicted value
- MaxSamples - determines the maximum number of data instances in a classhttps://github.com/

rest of them are used for version control.

### <a name="create-new-dataset"></a> Create new dataset.

To generate a new dataset we need to run the [generate.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/generate.py)

It would ask for a class label and runs until the maxSamples is reached. This would create a new csv file  
Each time You update a dataset a new version of csv is created 
```
python generate.py
```
Note: An incomplete Generation would end up in adding invalid data into the config file.

### <a name="train-new-dataset"></a> Train the new dataset.
Just like the dataset, each time you train a new version of ai model is created.

Feel free to Tweak the hyper parameters, an important parameter is epoch. The defualt value is 100 you may need to tweak it according to your data, but the defualt should work fine

Note: The data used for training will be the 
taken from config file ["CurrentData"]

Then you can start the training procedure by executing the [train.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/train.py)
file
```
python train.python
```
## <a name="how-it-works"></a> How it works.
