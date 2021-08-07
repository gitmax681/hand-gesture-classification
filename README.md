# Hand Gesture Recognition.

If you are looking for a quick hand gesture recognition model, you are at the right place.

In the repository I have created two models, a classic machine learning model and a neural network model. which would help you to get started with recognition of hand gestures within no time.

| id |      model      | average accuracy | file |
|----------|:-------------:|:---:|:-------:|
| 1 |  K Nearest Neighbour | 95.1%  | [KnnModel.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/KnnModel.py)
| 2 |  Neural Network Model  | 97.8%|   [neuralModel.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/KnnModel.py)

It is implemented using mediapipe and tensorflow. [more](#how-it-works)

> :warning: This is just for understanding basics. This model may not be memory efficient and effective.

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
- MaxSamples - determines the maximum number of data instances in a class. recommended not to go above 250

rest of them are used for version control.

### <a name="create-new-dataset"></a> Create new dataset.

To generate a new dataset we need to run the [generate.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/generate.py)

It would ask for a class label and runs until the maxSamples is reached. This would create a new csv file  
Each time You update a dataset a new version of csv is created 
```
python generate.py
```
> Note: An incomplete Generation would end up in adding invalid data into the config file.

### <a name="train-new-dataset"></a> Train the new dataset.
Just like the dataset, each time you train a new version of ai model is created.

Feel free to Tweak the hyper parameters, an important parameter is epoch. The defualt value is 100 you may need to tweak it according to your data, but the defualt should work fine

> Note: The data used for training will be the 
taken from CurrentData in config file

Then you can start the training procedure by executing the [train.py](https://github.com/gitmax681/hand-gesture-recognition/blob/master/train.py)
file

```
python train.py
```

This would create a new Model and update the config file.




## <a name="how-it-works"></a> How it works.

![hand coordinates](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

At the heart of the model is the mediapipe library from google. this computer vision library recognizes hands 
from the video stream and create coordinates for all 20 points as shown in the picture.

This 3d co-ordinate vectors are then converted into 2d x,y coordinates, and using the euclidean distance formula we get the distance between all 20 points.

![euclidean distance forumula](https://bit.ly/3CsZRN9)

This gives us a array of 400 values and this 400 data points represent the structure of hand.

This array is then normalized between 1 and 0 and given into a feed forward neural network which would give a softmax probabilty of being each class which is implemented in tensorflow. 


## <a name="disadvantages"></a> Disadvantages.

Since we have 400 of float32 data points for representing state of hand it, even 100 data instances would nearly make 1mb. this create a problem when we need to have more than a 20 symbols.

Cause this is for learning purposes it won't be much of a deal.

Happy Coding :heart: from Arjun Manjunath