# BrainApps

Repo for applications based on [brainflow](https://github.com/Andrey1994/brainflow)

## P300 Speller
P300 speller is based on [Event Related Potentials](https://en.wikipedia.org/wiki/Event-related_potential). I use TKInter to draw UI and [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) to perform classification
### Installation
* Install [Git LFS](https://git-lfs.github.com/) because I use it to save eeg data and pickled classifier
* Use Python 2.7 x64, and install packages from requirements.txt, to use Python 3 you will need to retrain the classifier(Pickle is not portable between different OSes and Python's versions)
[![Watch the video](https://farm8.staticflickr.com/7811/45713649104_1b32faa349_h.jpg)](https://youtu.be/Hf2cXCzRm80)

## GUI
Brainflow GUI is based on R Shiny package and provides simple UI to monitor EEG\EMG\ECG data
![image1](https://farm2.staticflickr.com/1842/30854740608_e40c6c5248_o_d.png)
