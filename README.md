# Face Tracker Neural Network Project

Welcome to the Face Tracker project! 🎉 In this project, I aim to developed a neural network capable of detecting faces in images and video streams. Buckle up and lets do some coding! 

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Creating Our Labels](#creating-our-labels)
- [Images and Labels](#images-and-labels)
- [Creating Our Model](#creating-our-model)
- [Testing the Model](#testing-the-model)
- [Loss and Optimization Functions](#loss-and-optimization-functions)
- [Training the Neural Network](#training-the-neural-network)
- [Making Predictions](#making-predictions)
- [Real-Time Testing](#real-time-testing)
- [Conclusion](#conclusion)

---
## Project Overview

In this project, we focused on building a neural network to detect faces within images and video streams using deep learning techniques. Our model is based on the VGG16 architecture and is designed to:

- **Classify images** (face or no face).
- **Predict bounding box coordinates** for detected faces.
- **Utilize advanced techniques** to enhance detection accuracy.
- **Allow real-time processing** through video capture.

---

## Getting Started

To get started with this project, you'll need to set up your environment. Ensure you have the following dependencies installed:

- **Python**: Make sure you have Python 3.x installed.
- **TensorFlow**: The primary library for building the model.
- **OpenCV**: For image processing and video capture.
- **Matplotlib**: For visualizing the results.

```bash
%pip install labelme tensorflow opencv-python matplotlib albumentations

```

---

## Creating Our Labels

In this section, we created our training, testing, and validation labels, prepping our data for the model. This step is crucial because:

- **Labels define the output** the model should learn to predict.
- **JSON** files containing bounding box information for faces were used.
- The loading function parses these JSON files and formats them appropriately.

```bash
# Creating a label loading function e.g., adding our bounding box
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

train_labels.as_numpy_iterator().next()

```

---

## Images and Labels

Here, we combined our images and their corresponding labels to ensure everything is in sync. This step ensures that:

- Each image has a corresponding label for effective training.
- Data integrity is maintained, preventing mismatched labels and images.

```bash
# Checking our partitions length 
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

```
---

## Creating Our Model

Finally, it's time to create our Deep Learning model using the Functional API. The VGG16 architecture was selected due to its proven effectiveness in image classification tasks. Key points include:

- Layer configuration: The architecture consists of convolutional layers, pooling layers, and fully connected layers.
- Transfer learning: We utilize pre-trained weights to enhance performance on our specific task.

Here's how we initiated our model:

```bash
# In this project we will be using the VGG16
vgg = VGG16(include_top=False)

```
---

## Testing the Model

Once the model was built, we ran a quick test to ensure it was working correctly. Testing is vital because:

- It verifies model architecture and functionality before training.
- Initial predictions give insight into model performance.

We tested the model with the following code:

```bash
facetracker = build_model()
facetracker.summary()

# Reminder: X is our images and y is our labels 
X, y = train.as_numpy_iterator().next()
classes, coords = facetracker.predict(X)
classes, coords

```
---

## Loss and Optimization Functions

To optimize our model, we defined a loss function and an optimizer. Understanding these components is essential because:

- Loss functions measure how well the model's predictions match the actual labels.
- Optimizers adjust the model parameters to minimize loss.

We implemented the Adam optimizer as follows:

```bash
# One way to go about finding out the LR is 75% of the original learning rate after each epoch.
batches_per_epoch = len(train)
lr_decay = (1. / 0.75 - 1) / batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

```
---

## Training the Neural Network

We created a custom model class and began training our neural network. This stage is crucial for model learning, involving:

- Feeding data: The model processes batches of training data.
- Backpropagation: The model updates weights based on loss gradients.
- Monitoring performance: We keep track of validation metrics.

Here's the custom model class we created:

```bash
class FaceTracker(Model): 
    def __init__(self, eyetracker, **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs): 
        # Training logic here...
        pass

    def test_step(self, batch, **kwargs): 
        # Testing logic here...
        pass

    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

```
**Training initiated with the following code:**

```bash
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

```
---

## Making Predictions

With the model trained, we tested it to evaluate its accuracy. Predictions are essential to understanding how well the model has learned. Here’s how we did it:

- We processed test samples and generated predictions.
- Visualized results with bounding boxes to indicate detected faces.

Here’s the code that accomplishes this:

```bash
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20, 20)) 
for idx in range(4): 
    sample_image = test_sample[0][idx].copy()  # Make a writable copy
    sample_coords = yhat[1][idx]
    
    # Draw rectangle if the confidence score is above 0.9
    if yhat[0][idx] > 0.9:
        cv2.rectangle(
            sample_image, 
            tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
            tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)), 
            (255, 0, 0), 2
        )
    
    ax[idx].imshow(sample_image)

```
---

## Real-Time Testing

Finally, we tested our model in real-time using our camera! This feature allows users to see the model's effectiveness in action. Key points include:

- Continuous video capture: The model processes each frame in real time.
- Bounding box rendering: Detected faces are highlighted in the video stream.

Here’s how we implemented this:

```bash
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500, :]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    
    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Draw rectangles and labels...
        pass
    
    cv2.imshow('Face Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

```
---

## Conclusion
And there you have it!  Our Face Tracker model is ready to identify faces with flair! This project not only improved our understanding of neural networks but also equipped us with skills in data preparation, model building, and real-time application.

