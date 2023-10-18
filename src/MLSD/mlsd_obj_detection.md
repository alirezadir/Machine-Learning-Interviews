## 2D object detectors
### Two stage detectors
Two-stage object detectors are a type of deep learning model used for object detection tasks. These models typically consist of two main stages: region proposal and object classification.

* In the first stage, the region proposal network (RPN) generates a set of potential object bounding boxes within an image. These proposals are generated based on a set of anchor boxes, which are pre-defined boxes of various sizes and aspect ratios that are placed at different positions within the image. The RPN uses convolutional neural networks (CNNs) to predict the likelihood of an object being present within each anchor box and refines the coordinates of the proposal box accordingly.

* In the second stage, the object classification network takes the proposed regions from the RPN and classifies them into different object categories. This stage involves further processing of the region proposals, such as resizing them to a fixed size and extracting features using a CNN. The features are then fed into a classifier, typically a fully connected layer followed by a softmax activation function, to predict the object class and confidence score for each proposed region.

Two-stage object detectors, such as Faster R-CNN and R-FCN, are known for their high accuracy and robustness in object detection tasks. However, they can be computationally intensive due to the need for both region proposal and object classification, and can be slower than single-stage detectors.

### One stage detectors 
One-stage object detectors are a type of deep learning model used for object detection tasks. These models differ from two-stage detectors in that they perform both region proposal and object classification in a single step.

The most popular one-stage detector is the YOLO (You Only Look Once) family of models. The YOLO model divides the input image into a grid of cells, and each cell predicts bounding boxes, objectness scores, and class probabilities for objects that appear in that cell. The objectness score represents the likelihood that the cell contains an object, and the class probabilities indicate the predicted class of the object.

Other one-stage detectors, such as SSD (Single Shot Detector) and RetinaNet, use a similar approach but with different architectures. They typically use a series of convolutional layers to extract features from the input image and generate a set of anchor boxes at various scales and aspect ratios. The network then predicts the likelihood of an object being present within each anchor box, and refines the box coordinates accordingly.

One-stage detectors are known for their speed and efficiency, as they can perform both region proposal and object classification in a single forward pass. However, they may not be as accurate as two-stage detectors, especially for small or highly occluded objects.

### Metrics 
* Precision 
  * calculated based on IOU threshold
* AP: avg. across various IOU thresholds 
* mAP: mean of AP over C classes 