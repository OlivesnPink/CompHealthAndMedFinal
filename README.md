# Environment
Ran on the UML edu gpu servers, setup may break on other systems/architectures
## Setting up the environment
Make sure you have conda or miniconda installed, this should let you quickly set up the environment for usage. The provided environment has most if not all relevant packages installed such as matplotlib, pandas, pytorch, and so on. To import and use the environment, run the following commands <br>
`python3.12 -m venv <path to virtual environment>`<br>
`source <path to virtual environment>/bin/activate`<br>
`pip install -r ./requirements.txt`

## Updating environment
If any new packages need to be installed, make the requirements file is updated before pushing an update to the repository. Just add to module to requirements.txt and make sure to commit and push it.

# GIT practices
You can find a git cheat sheet [here](https://git-scm.com/cheat-sheet)<br>
Make sure when making any updates, you create a new branch, push the branch to remote/github, and then create a merge/pull request to the main branch

# Dataset
Please do not add the dataset to the repository as it is too big. <br>
The dataset can be found [here](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid). <br>
You need an IEEE account to download the dataset, use the all classes dataset rather than the challenge dataset.

## Compressing the dataset

Since the dataset is too big to fit on the UML gpu servers, a script has been attached (DatasetCompresser.ipynb), run the notebook file specifying the target path, will result in all the images being shrunken down and rescaled to 1028x1028. Once it's been compressed it can be transferred to the GPU servers using filezilla

# Resources

### RFMiD: Retinal Image Analysis for multi-Disease Detection challenge
Link to paper [here](https://www.sciencedirect.com/science/article/pii/S1361841524002901)<br>
paper goes over the challenge and what competitors did to get their results, could provide useful ideas on how to improve models