# Toronto Traffic Accident Injury Severity Analysis
## _Predicting injuries Using Machine Learning_
The Toronto Police Open Data portal gives us an opportunity to analyze reports from law enforcement in Toronto, and make predictions on the resulting information. An example of this data, detailing traffic incidents between 2007 and 2017 was uploaded to [Kaggle](https://www.kaggle.com/jrmistry/killed-or-seriously-injured-ksi-toronto-clean), and with it we created a machine learning model that could predict what severity of injury a party in a traffic accident would receive.

This process involved cleaning the data of issues such as redundant and misleading fields, determining which fields were most relevant to the result using truncated Singular Value Decomposition (SVD) and logistic regression, creating a random forest model for the classification, and improving our results with RandomSearchCV.

## Development

To contribute to this project, please complete the following steps:
- `git checkout main`
- `git pull main`
- `git checkout -b <your_branch_name>` (create a branch name based on the changes you want to make)
- Make your changes
- Before committing your file, please reset your runtime to get rid of the runtime numbers (IE the numbers between the square brackets beside each cell). The option should be under runtime -> restart runtime, or something like that.
- Save your file - both the .ipynb and the .py
- `git add ksi_analysis.ipynb`
- `git add ksi_analysis.py`
- `git commit -m "Your commit message here"` - please write a short but descriptive message!
- `git push origin <your_branch_name>`
- Open a Pull Request (PR) in Github, attempting to merge your branch into main, and request reviews from the group
- When you have approvals, merge your pull request

## Authors
- [Daniel Siegel](https://github.com/danielmaxsiegel) - 101367445
- [Michael McAllister](https://github.com/michaeldavidmcallister) - 101359469
- [Hom Kandel](https://github.com/homnath008) - 101385341
- [Eduardo Bastos de Moraes](https://github.com/eduardomoraes) - 101345799
