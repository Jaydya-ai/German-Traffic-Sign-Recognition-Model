## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- Required Python packages:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `opencv-python`
  - `scipy`
  - `tqdm`

You can install these packages using pip:

```bash
pip install numpy matplotlib pandas opencv-python scipy tqdm
```

## Datasets
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

To run the script you will need GTSRB_Final_Test_HueHist.zip, GTSRB_Final_Training_HueHist.zip, GTSRB_Final_Training_HOG.zip, GTSRB_Final_Test_HOG.zip, GTSRB_Final_Test_GT.zip, GTSRB_Final_Test_Images.zip and GTSRB_Final_Training_Images.zip.

Ensure the GTSRB datasets are placed in the directory structure shown below:

```bash
analyse_data.py
GTSRB/
    ├── Final_Training/
    │           └── Images/
    │           └── HOG/
    │           └── HueHist/  
    └── Final_Test/
    │           └── Images/
    │           └── HOG/
    │           └── HueHist/
    └──GT-final_test.csv
```

## Running the Script
To execute the script, navigate to the directory containing analyse_data.py and the datasets, and run the command:

```bash
python analyse_data.py
```