
# ML-Zoomacamp Capstone Project
## Exploring Covid-19 Image Dataset

### About Dataset

Dataset was taken from Kaggle Competition

https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data

**Context**

COVID-19 (coronavirus disease 2019) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a strain of coronavirus. The first cases were seen in Wuhan, China, in late December 2019 before spreading globally. The current outbreak was officially recognized as a pandemic by the World Health Organization (WHO) on 11 March 2020.

**Content**

Dataset is organized into 2 folders (train, test) and both train and test contain 3 subfolders (COVID19, PNEUMONIA, NORMAL). DataSet contains total 6432 x-ray images and test data have 20% of total images.

Directory structure scheme for the Covid 19 dataset:
```
Covid19-dataset/
│
├── Train/
│   ├── Covid/
│   │   ├── img1.jpg
│   │   ├── img2.jpg 
│   │   └── ... (more images)
│   ├── Normal/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ... (more images)
│   └── Viral Pneumonia/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ... (more images)
│
└── Test/
    ├── Covid/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ... (more images)
    ├── Normal/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ... (more images)
    └── Viral Pneumonia/
        ├── img1.jpg
        ├── img2.jpg
        └── ... (more images)
```
### CNN Model Architecture and Evaluation
In this project, was used a Convolutional Neural Network (CNN) architecture to classify X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal.

To select the best model, it was used the following evaluation metrics:

Accuracy Score: To measure the overall performance of the model.

Precision: To evaluate how many of the positive predictions were correct.

Recall: To evaluate how many of the actual positive cases were correctly identified.

Confusion Matrix: To visualize the performance of the classification model and the distribution of errors.

Based on the evaluation metrics, it was selected the best-performing model to make predictions on the test data.

Image Augmentation techniques were used during training. These techniques included random rotations, shifts, flips, and zooms, which artificially increased the diversity of the training dataset.

Additionally, VGG16, a popular deep learning model pre-trained on the ImageNet dataset, was used as a feature extractor. The model's convolutional layers were used to extract features from the X-ray images before feeding them into the custom CNN layers for final classification.


### How to Set Up the Project

Follow these steps to set up the project from GitHub on your local machine:

#### 1. Clone the Repository

First, clone the repository from GitHub to your local machine. Open a terminal or command prompt and run:
```
git clone https://github.com/OleksiiLatypov/mlzoomcamp-capstone01.git
```

#### 2. Navigate to the Project Directory

Change to the project directory:
```
cd mlzoomcamp-capstone01
```

#### 3. Set Up the Virtual Environment And Install Pipenv
For Windows:
```
python -m venv venv .\venv\Scripts\activate
```
For macOS/Linux:
```
python3 -m venv venv source venv/bin/activate
```
Then install Pipenenv:
```
pip install pipenv
```

#### 4. Install Project Dependencies

Once the virtual environment is activated, install the required dependencies using Pipenv (which is used to manage the project dependencies):

```
pipenv shell
```

you can install all the dependencies from Pipfile by running pipenv install

or manually:

```
pipenv install tensorflow keras pandas matplotlib numpy seaborn scikit-learn gunicorn flask
```

#### 5.Install Docker
- Download & Intall Docker Desktop https://www.docker.com/
- Build a Docker image from a Dockerfile in the current directory:
```
docker build -t covid-prediction .
```
- Run it, execute the command below:
```
docker run -it -p 9696:9696 covid-prediction:latest
```
#### 6. Run app and make prediction

- Open new terminal and run:
```
python predict_test.py
```

For uncommented link in ```predict_test.py``` file:

https://www.princeton.edu/sites/default/files/styles/scale_1440/public/images/2020/05/x-ray-image-2b_full.jpg?itok=2FO93vqG

It should be next prediction:

{'prediction': 'Covid', 'probability': 0.9971820116043091}
