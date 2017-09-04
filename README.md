# Dog_vs_Cat_Kaggle_Dataset

This is an approach to distinguish between cats and dogs described <a href=”https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/”>here</a>.
To run this code, first download the `train` and `test` dataset from this <a href=”https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data”>link</a>. 
You need have an account in <a href=”https://www.kaggle.com/”>Kaggle</a>.Then run 'train.py'.I used 24000 images for training and 1000 images for testing. 

Final accuracy on test set was `0.7857`. You can improve it by adding more layers, using dropout and varying number of neurons at each layer.
I have provided my models files in `model_files` folder. You can use that for predicting over an image. 
