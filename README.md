# The Feedback Filer: Topic Modeling of Product Reviews

## Overview
The goal of this project is to utilize Natural Language Processing (NLP) in conjunction with unsupervised learning to classify consumer feedback by topic, enabling developers and designers to streamline the optimization of their product. Ideally, this model would be used to supplement user research teams by not only confirming human findings, but also capturing insights that the team may have missed, overlooked, or deemed unimportant.

## Project Organization
    ├── README.md             <- The top-level README for developers using this project
    │
    ├── LICENSE               <- Copyright and permissive license
    │
    ├── Notebooks             <- Project source code in Jupyter notebook
    │   └── project4.py       <- Script with helper functions
    │
    ├── Reports               <- Various reports
    │   ├── proposal.pdf      <- Project proposal
    │   ├── summary.pdf       <- Project summary
    │   └── presentation.pptx <- Project presentation slide deck
    │
    ├── Flask App             <- Web application materials
    │   ├── src               <- Source code for api and app
    │   ├── images            <- Miscellaneous images
    │   ├── model             <- Model weights saved as pkl
    │   ├── static            <- Flask static files
    │   └── templates         <- HTML web template
    │   

## Data
Product review data was obtained via scraping G2 and Capterra, two popular consumer review websites, where I obtained reviews from nearly 200 individuals. This pilot study focused on the online product reviews regarding a Point-of-Sale system (POS) named Aloha. 

## Tools
* ***Web Scraping:*** BeautifulSoup
* ***Data Processing:*** Pandas, Re, TextBlob, NLTK
* ***Modeling:*** Scikit-learn
* ***Data Visualization:*** Matplotlib
* ***App Development:*** Flask
* ***Presentation:*** Powerpoint   

## Deliverables
With the resulting model, I was able to put together a working prototype of what my resulting product would look like using Flask. The program would enable users to submit transcripts from interviews, focus groups, usability tests, diary studies, and participatory design workshops, etc., leaving the model to cluster the data, then group it with its corresponding cluster. The product is intended to supplement user research teams in analyzing user research data, enabling them to give quick and actionable insights towards optimizing their products. A screenshot of the resulting project is illustrated below.
  

![Flask Application](https://github.com/broco347/feedback-filter/blob/master/Flask_App/images/Screen%20Shot%202019-07-08%20at%2015.44.25.png)

## LICENSE
MIT © Brooke Ann Coco
