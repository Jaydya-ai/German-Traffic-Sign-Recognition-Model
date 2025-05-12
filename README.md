# German-Traffic-Sign-Recognition-Model
This project implements deep learning models to classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It includes preprocessing, training, evaluation, and comparison of CNN-based architectures.

## 📁 Project Structure

- `GTSRB-Code.ipynb`: Main Jupyter Notebook containing all steps: data preprocessing, model training, evaluation.
- `Final_Training/Images/`: Directory expected to contain subfolders of traffic sign images and their corresponding annotation CSV files (e.g., `GT-00000.csv`).
Ensure the dataset is extracted into the following directory structure:

```bash
GTSRB-Code.ipynb
Final_Training/
    └── Images/
        └── [subfolders with traffic sign images]
        └── GT-00000.csv (annotation files)
```

## 🧠 Models Implemented

- **Deep CNN**
- **Simple CNN**
- **LeNet**
- **MobileNetV2**

These are trained and evaluated using accuracy, classification report, and confusion matrix.

## 🛠️ Requirements

Install the following Python libraries:

```bash
numpy
pandas
matplotlib
scikit-learn
opencv-python
tensorflow
```

## 🧪 Usage

1. Download the GTSRB dataset from [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
2. Extract the `Final_Training/Images` directory into the same folder as the notebook.
3. Open `GTSRB-Code.ipynb` and run all cells.

## 📊 Evaluation

- The models are evaluated using:
  - Accuracy on the validation set
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

## 📌 Notes

- Images are resized to **32x32 pixels**.
- Preprocessing includes **histogram equalization** for improved contrast.
- Training includes **early stopping**, **learning rate reduction**, and **data augmentation**.

## 📷 Sample Visualization

The notebook also includes visualizations of:
- Sample traffic sign images
- Augmented images
- Model performance metrics

## 🔒 License

This project is for educational and research purposes only.
