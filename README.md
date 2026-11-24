# AI Handwritten Calculator

An interactive handwritten calculator powered by a Dense Neural Network built with Keras and TensorFlow, paired with a real-time drawing interface in Tkinter. 

Unlike standard digit-only classifiers, this model recognizes both **digits (`0`–`9`)** and **mathematical operators (`+` and `-`)**, allowing users to hand-draw entire equations and compute results on the fly. It also features **live incremental learning**, enabling the model to learn from its mistakes directly through the GUI.

---

## Features

* **Multi-Class Handwritten Recognition:** Classifies 12 distinct classes (`0–9`, `+`, `-`) from 28×28 grayscale images.
* **Interactive Drawing Canvas:** Draw digits and operators smoothly using a Tkinter GUI backed by Pillow (`PIL`) image processing.
* **Real-Time Equation Builder & Solver:** Sequentially recognizes drawn characters, builds the mathematical expression, and evaluates the final result.
* **Live On-Device Correction (`train_on_batch`):** If the model misclassifies a symbol, click the corresponding correction button to immediately update the neural network's weights and save the improved model.
* **Automatic Color Normalization:** Automatically detects and inverts black-on-white strokes to match the white-on-black MNIST format.

---

## Model Architecture & Training

The model uses a lightweight, fast **Dense Neural Network** trained on a combined dataset:
1. **MNIST Dataset:** 60,000 handwritten digit images (`0–9`).
2. **Handwritten Math Symbols Dataset (Kaggle):** Up to 4,000 samples each for `+` (class `10`) and `-` (class `11`), preprocessed and resized to 28×28 pixels.

**Network Structure:**
* **Input Layer:** Flattened 784-pixel vector (`28 × 28` normalized to `[0.0, 1.0]`).
* **Hidden Layer:** `Dense(64, activation='relu')`.
* **Output Layer:** `Dense(12, activation='softmax')`.
* **Optimizer & Loss:** `adam` with `sparse_categorical_crossentropy`.

---

## Project Structure

```text
├── Modèle_IA_V2.py                 # Dataset preprocessing, model training, and saving
├── Interface.py                    # Tkinter GUI, live inference, and incremental learning
└── mon_modele_IA_keras_V2.keras    # Saved Keras model (generated after training)
```

---

## Installation & Usage

### 1. Prerequisites
Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install tensorflow keras numpy pillow kagglehub
```
*(Note: `tkinter` is included with most standard Python installations).*

### 2. Train the Model
Run the training script to download the math symbols dataset, combine it with MNIST, and generate `mon_modele_IA_keras_V2.keras`:

```bash
python Modèle_IA_V2.py
```

### 3. Launch the Calculator
Once the model file is saved in the project directory, start the interactive interface:

```bash
python Interface.py
```

### 4. How to Use the Interface
1. **Draw** a digit (`0–9`) or an operator (`+`, `-`) on the white canvas using your mouse.
2. Click **Valider** to run inference and append the recognized symbol to the current equation (displayed in the console).
3. *(Optional)* If the prediction is wrong, click the matching button under **CORRECTION** (`0–9`, `+`, or `-`). The model will train on that sample immediately and overwrite the saved model file.
4. Click **Effacer** to clear the canvas before drawing the next character.
5. Click **Calculer** once your equation is complete to evaluate and display the result.
