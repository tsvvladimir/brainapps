## P300 Speller
P300 speller is based on [Event Related Potentials](https://en.wikipedia.org/wiki/Event-related_potential). I use TKInter to draw UI and [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) to perform classification
### Installation
* Electrode placement: P3,P4,C3,C4,T5,T6,O1,O2
* Install [Git LFS](https://git-lfs.github.com/) because I use it to save eeg data and pickled classifier
* Use Python 2.7 x64, and install packages from requirements.txt, to use Python 3 you will need to retrain the classifier(Pickle is not portable between different OSes and Python's versions)
[![Watch the video](https://farm8.staticflickr.com/7811/45713649104_1b32faa349_h.jpg)](https://youtu.be/1GdjMx5t4ls)
