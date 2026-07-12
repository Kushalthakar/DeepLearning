# **Q/A**



###### Why is a fully connected network a poor choice for CIFAR-10?

This dataset image are very small (32 x 32) pixels only and this fully connected treats each pixel separately. CNN is better because it learns local pattern like edge, texture and shape.



###### What if the dataset were heavily imbalance?

if it is imbalance then the accuracy could be misleading.



###### Why use a virtual environment?

A virtual environment keep dependencies separate and can easily run on another machine



###### Why increase filters in deep layer?

It helps CNN learn more complex features.



###### How do pooling layers helps and what is lost?

Pooling layer make model more robust to small shifts by keeping the strongest feature in a small region but at the same time i also loses some extra location and details.



###### Why use exactly 10 SoftMax output units?

It need 10 units because it CIFAR - 10 has 10 classes. It is used because each image belongs to one class.



###### Why normalization input?

Normalized input because smaller values make weights update smoother. Scaling value from 0-255 to 0-1 helps the optimizer learn fast.



###### How does the learning rate affect training?

The learning rate controls how big each weight update is. If it is too high, training may became unstable and the loss may jump up and down.



###### What patterns show overfitting or underfitting?

Overfitting happens when training accuracy keeps increasing but validation accuracy stops improving or decrease. Underfitting happens when both training and validation accuracy stays low and flat, meaning the model is not learning enough.



###### How would you choose a sensible number of epochs and batch size?

A sensible number of epochs can be chosen by watching the validation accuracy and validation loss. Batch size is usually chosen based on memory and training stability.



###### What can a confusion matrix reveal that accuracy cannot?

A confusion matrix shows which classes the model predicts correctly and which classes it confuses. A confusion matrix shows class-level mistakes.



###### Why verify that a reload model gives the same prediction ?

Verifying the reload model ensures that the saved file works correct and that the model ca be reused later without retraining.



###### Why apply data augmentation only to training data ?

It should only be applied to training data because it helps the model learn from varied examples.



###### How do you know if the adding dropout improved the model?

By comparing the training and validation curves before and after padding. If validation accuracy improves, validation loss decreases, or the gap between training and validation accuracy becomes smaller, then dropout likely helped reduce overfitting



###### Why is documented experimentation more important than one high accuracy number?

A single high accuracy number does not explain why model worked, while experiments show your reasoning improvements, and learning process.

