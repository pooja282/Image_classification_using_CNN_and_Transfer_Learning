Transfer Learning

Steps:

1.Preprocess Data: The images are converted from RGB to BGR, then each color channel is zero-centered .
2.Loading of the pre-trained network.
3.Freeze the convolutional base before compiling and train the model.
4.Define a FNN architecture.
5.Callbacks.
6.Compile and train.




# Image_classification_using_CNN_and_Transfer_Learning

Classification Report for VGG16  : 

                precision    recall  f1-score   support

           0       0.88      0.90      0.89      1541
           1       0.95      0.92      0.94      1442
           2       0.99      0.99      0.99      1519
           3       0.88      0.94      0.91      1497
           4       0.94      0.88      0.91      1501

    accuracy                           0.93      7500
   macro avg       0.93      0.93      0.93      7500
weighted avg       0.93      0.93      0.93      7500




VGG16 Model
	
![image](https://github.com/user-attachments/assets/b9a3f88a-6843-4c8f-a63f-c1f6b2c5c87c)

 


![image](https://github.com/user-attachments/assets/17011ea8-0e4f-499c-ad45-3fc27ec21f99)





![Screenshot (3)](https://github.com/user-attachments/assets/f83cf123-7139-484b-b13b-afec486a5b1c)




