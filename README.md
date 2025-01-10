
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


