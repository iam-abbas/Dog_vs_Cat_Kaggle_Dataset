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
