# Dogs vs. Cat Kaggle Dataset

<b>Requirements:</b>
1. Python 3
2. Tensorflow
3. Numpy
4. Tqdm

This is an approach to distinguish between cats and dogs described <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/"> here</a>.
To run this code, first download the `train` and `test` dataset from this <a href="https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data"> link</a>.
You need have an account in <a href="https://www.kaggle.com/"> Kaggle</a>. Then run `train.py`.I used 24000 images for training and 1000 images for testing. 

Final accuracy on test set was `0.7857`. You can improve it by adding more layers, using dropout and varying number of neurons at each layer.

I have provided my models files in the `model_file` folder. I had to split the file `DogvsCat_model.ckpt.data-00000-of-00001`into two parts because github doesn't allow files larger than 25 mb. You can join them by running```cat DogvsCat_model.ckpt.data-00000-of-00001.part* > DogvsCat_model.ckpt.data-00000-of-00001``` on terminal. These model files can be used for predicting over an image. 

I wrote the script `predict.py` to predict over test images using my model and save the results in `Submission.csv` file. It gave me a logloss score of `10.69341`. 

I wrote another script `predict_inception.py` to predict images using <a href="https://www.kaggle.com/google-brain/inception-v3"> Inception v3</a> model file. But it couldn't find any dog or cat in over 4000 images. I used my former model to predict on those images and finally merged these predictions. That gave me a logloss score of `1.79879`
