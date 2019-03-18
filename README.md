# Face-Recognition
> Face Recognition notebook with Face Detection and crop of image <br>
> All the code was written in colab's notebook <br>
> The idea and the <b>trained model</b> used are from: [deeplearning.ai](https://www.deeplearning.ai/) courses

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
