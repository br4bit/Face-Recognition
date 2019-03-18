# Face Recognition with Detection and Crop
> Face Recognition notebook with Face Detection and crop of image <br>
> All the code was written in colab's notebook <br>
> The idea and the <b>trained model</b> used are from: [deeplearning.ai](https://www.deeplearning.ai/) courses
## How it Works
![Example0](https://i.gyazo.com/a33c2bd34c2c7f5a6790862f64c6806f.gif)
![Example3](https://i.gyazo.com/e1cd4312fc6f90a8309de812f7689fef.png)
## Getting Started
## Installing
Setup on Google Drive and use Colaboratory
* Download github repository
```
$ git clone https://github.com/br4bit/Face-Recognition.git
```
Upload on Google Drive <br>
### Running test
#### Step 1 : First we need to load some images of people in ``` images ``` directory like in this example:
> images/
>> John
>>> img1.jpg,img2.jpg,img3.jpg,ecc.

The code cell will load all images in directory (all formats are accepted).<br>
_Note: More images will improve accuracy of algorithm.<br>
it is advisable to use 10 images per person, taken in different light and ambient conditions._
#### Step 2 : Open Face_Recognition.ipynb with Colaboratory. Execute the first two cells code.
>![Example](http://g.recordit.co/KiuqIc0Hfa.gif)
#### Step 3 : Execute the code cell that load all images in a database as Name -> Image. 
If in the database there are two images of the same person, we will see two vectors for the same person. <br>
It's a Map<Key, Element> when multiple items has the same key.
![Example2](http://g.recordit.co/eac08bqaSG.gif)
#### Step 4 (Optional): There is a cell code that take photo from camera of phone/pc.
![Example4](https://i.gyazo.com/2c540ffb92d1cc204340a2d5f679a24d.png)
#### Step 5: The final exciting step, this cell code will recognize face captured from ``` camera.jpg ```.
![Example6](https://i.gyazo.com/d5c280680e99661a6f317d9d6804ee9a.png)
It's a 1:N Face Recognition Problem. The function ``` who_is_it() ``` will return the distance computed by triplet loss function.
![Example5](https://i.gyazo.com/8709bed21d3e820883989b824abce3e1.png)

## Built With
* [![Made In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
* [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
* [IPython](https://ipython.org/)
* [TensorFlow-Keras](https://www.tensorflow.org/)
* [OpenCV Library](https://opencv.org/)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Acknowledgments
Imagination
