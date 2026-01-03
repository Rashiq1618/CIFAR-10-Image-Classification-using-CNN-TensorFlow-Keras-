## CIFAR-10 Image Classification using Deep CNN
This repository implements a **high-accuracy Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset** using **TensorFlow/Keras**.
It includes scripts for **training the model** and **running inference on custom images**.

---
## Training Accuracy

The following figure shows the training and validation accuracy and loss
over multiple epochs on the CIFAR-10 dataset.

<p align="center">
  <img src="accuracy.jpeg" width="800">
</p>

## CIFAR-10 Classes

```
Plane, Car, Bird, Cat, Deer,
Dog, Frog, Horse, Ship, Truck
```

---

## Project Structure

```
.
├── train_model.py          # Train the CNN model
├── run_model.py            # Run inference on a custom image
├── cifar10_super_model.h5  # Trained model (generated after training)
├── deer.jpg, car.jpg etc   # Sample input images
├── README.md               # Project documentation
```

---

## Installation

Clone the repository


Install required dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

---

## Train the Model

Run the training script:

```bash
python train_model.py
```

During training:

* CIFAR-10 dataset is downloaded automatically
* Data augmentation is applied
* Model is trained for up to 50 epochs
* Trained model is saved as `cifar10_super_model.h5`

---

## Run Inference

1. Place a **32×32 RGB image** in the project directory
2. Update the image filename in `run_model.py` if necessary
3. Run:

```bash
python run_model.py
```

### Example Output

```
Prediction is: Deer
```

The image will be displayed using Matplotlib.

---

## Image Input Requirements

| Property      | Value                        |
| ------------- | ---------------------------- |
| Image size    | 32 × 32 pixels               |
| Channels      | RGB (3)                      |
| Normalization | Pixel values in range [0, 1] |


---

## Notes

* The model is trained on **small 32×32 CIFAR-10 images**
* Performance may be limited on real-world, high-resolution images




