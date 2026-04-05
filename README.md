# Environment
Ran on the UML edu gpu servers, setup may break on other systems/architectures
## Setting up the environment
Make sure you have conda or miniconda installed, this should let you quickly set up the environment for usage. The provided environment has most if not all relevant packages installed such as matplotlib, pandas, pytorch, and so on. To import and use the environment, run the following commands <br>
`python3.12 -m venv <path to virtual environment>`<br>
`source <path to virtual environment>/bin/activate`<br>
`pip install -r ./requirements.txt`

## Updating environment
If any new packages need to be installed, make the requirements file is updated before pushing an update to the repository. Export the environment after adding any packages by using the following command <br>
`pip freeze > requirements.txt`

# GIT practices
You can find a git cheat sheet [here](https://git-scm.com/cheat-sheet)<br>
Make sure when making any updates, you create a new branch, push the branch to remote/github, and then create a merge/pull request to the main branch

# Dataset
Please do not add the dataset to the repository as it is too big. 