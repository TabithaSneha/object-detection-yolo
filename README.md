# Object Detection with YOLO

***

In this project, the YOLO algorithm has been implemented on the PASCAL VOC Dataset for the purpose of object detection of real-time objects.

## Introduction:-

Object Detection is a Computer Vision technology that deals with the detection and localization of objects in an image belonging to a predefined set of classes. It is an advanced form of image classification where a neural network is trained to predict objects in an image, in the form of bounding boxes.
YOLO (You Only Look Once) is a state-of-the-art algorithm for Object Detection that makes use of Convolutional Neural Networks to detect objects in real-time. It is popular because of its speed and accuracy. In a single forward propagation through the CNN, it outputs the predictions of all the class probabilites and bounding box parameters of an image simultaneously. And hence YOLO!

### Working of the YOLO Algorithm:

* **Residual Boxes**: The input image is first split into SxS grids.
* If the object's midpoint appears in a grid cell, then that cell will be responsible for detecting it.
* Each grid cell outputs a prediction with B Bounding Boxes and provides their corresponding confidence scores.
* The Bounding Box is in the form of [x, y, w, h], where (x, y) are the coordinates of the object midpoint in the cell relative to the cell and (w, h) is the width and height of the object also relative to the cell.
* The ouput predictions of each cell are encoded as follows: [S x S x (5*B + C)], where C is the number of class probabilities.
* For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2 (for Wide and Tall Bounding Boxes). PASCAL VOC has 20 labelled classes so C = 20. Thus, the final prediction is a 7 × 7 × 30 tensor.
* **Intersection Over Union**: A popular metric to measure localization accuracy by dividing the intersecting area of the bounding box and ground truth box with the area of its union. Larger the IOU, better is the prediction.
*  **Non Maximal Suppression**: Since multiple cells can contain one object and each of them would predict their bounding boxes for it, NMS threshold is used to suppress those bounding boxes with lower confidence scores.
*  **Model Architecture of YOLO v1**: The detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 convolutional layers reduce the features space from preceding layers. The size of the input image is 448 x 448 x 3, which eventually comes down to 7 x 7 x 30.
![Pic](https://i.imgur.com/oyBoc1y.png)
Image Source: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* **Loss Function**: The YOLO v1 Loss Function is divided into 4 parts-
  * Box Loss: Mean squared error of the predictions of the coordinates of midpoint (x, y) and the width and height of the bounding box (w, h).
  * Object Loss: Mean squared error of the probability that an object is present in the cell or not.
  * No Object Loss: Mean squared error of the probabilty that an object is not present in the cell.
  * Class Loss: Mean squared error of the predictions of the class probabilities of the object.
   <img src="https://i.imgur.com/lOB3j7E.png" height="400" width="600" >

  Image Source: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* **Mean Average Precision**: The Average Precision (AP) is calculated as the area under the Precision-Recall curve for a set of predictions per class. The average of this value taken over all classes called mean Average Precision (mAP) is calculated.

### Libraries used:-

* PyTorch
* Numpy
* Matplotlib
* Pandas

### Dataset used:-

The PASCAL VOC Dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2).

### Parameters used:-



| Parameters | Values |
| -------- | -------- |
| Learning rate     | 1e-4     |
| Batch Size     | 16     |
| Weight Decay     | 5e-4     |
| Epochs     | 200     |
| Optimizer     | Adam     |

## Results:-

![Pic](https://i.imgur.com/M5QnDnm.jpg)

As can be seen, YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that the model can predict. It struggles with small objects that appear in groups, such as flocks of birds or crowds of people.

| Parameter | Result | 
| -------- | -------- | 
| Train mAP     | 0.98536     |
| Mean Loss     | 5.292     | 

## Further Work on this Project:-

Currently working on the YOLO v3 implementation of Object Detection on the PASCAL VOC Dataset to train the model to a decent level of accuracy and be able to visualise the output.

## References:-

* [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* [YOLO v1 from Scratch by Aladdin Persson](https://m.youtube.com/watch?v=n9_XyCGr-MI)
* [YOLO: Real-Time Object Detection Explained](https://www.v7labs.com/blog/yolo-object-detection)
* [Introduction to YOLO Algorithm for Object Detection](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)
