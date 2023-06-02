# Replication study for "Application of deep and machine learning techniques for multi-label classification performance on psychotic disorder diseases",by Elujide et al, 2021

The original paper used two separate environments for its experiments: A Google Colab Notebook running Python 3.6 and a local installation of Python 3.7 running on a Windows 10 workstation.

The Goole Colab environment was used to run the Keras TensorFlow DNN experiments, while the local Python 3.7 environment was used to run experiments on ML algorithms in the scikit-learn library. 

## 1 Citation

Israel Elujide, Stephen G. Fashoto, Bunmi Fashoto, Elliot Mbunge, Sakinat O. Folorunso, and Jeremiah O. Olamijuwon. 2021. Application of deep and machine learning techniques for multi-label classification performance on psychotic disorder diseases. Informatics in Medicine Unlocked, 23:100545.

## 2 Dependencies

### 2.1 Local Python 3.7 Windows 10 Environment:
	
	Install python 3.7 for this project. Install all dependencies using: 
	
		pip3 install -r requirements.txt

### 2.2 Google Colab setup

	Open the dnn_experiment.ipynb file in Google Colab. Run the first cell in the notebook and enter "1" when prompted, to select Python 3.6. 
	After installation of Python 3.6 completes, you can select "run all" to execute all cells. 

	
## 3 Data download instructions

	The data can be acquired from the following link: https://ars.els-cdn.com/content/image/1-s2.0-S2352340917303487-mmc2.csv
	However, the data is also stored in this github repository and will be automatically downloaded by the pre-processing code in both the Python 3.7 project and the Google Colab Notebook. 


## 4 Code Execution

### 4.1 Python 3.7 Project Execution

	The python project which executes the scikit-learn experiments is contained in the files:

	main.py
	ml_multilabel_experiment.py
	ml_singlelabel_experiment.py
	preprocessing.py

	Run main.py to execute all experiments. The dataset will automatically be downloaded from the github repository. Results from each experiment, along with the processing times, will be displayed to the console.

### 4.2 Google Colab Notebook Execution

	Run all cells, or each cell individually, in-sequence. First, the pre-processing code will download the data and create local copies of the balanced and imbalanced datasets. 
	Subsequent experiments will utilize these files. Each experiment will display its results, along with the processing times, as well as plotting loss and accuracy curves where appropriate.
	
	Note that for some of the long-running blocks of code, such as the ablations, you may have to respond to the "I am not a robot" prompt that Google Colab surfaces after several minutes of processing time. 
	If Colab does not receive a response, it will terminate the session.
