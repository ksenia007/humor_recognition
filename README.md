# Humor Project

This is a repository for Princeton COS598C final project, Spring 2020, taught by Prof. Danqi Chen. 

👉 In short, the project explored the performance of frozen/unfrozen pre-trained `BERT-base-uncased` models on the joke recognition, and the effect dataset formation has on the performance. 

🧠 Obtained a good performance for joke recognition. Observed that cleaner datasets give better results and adding lower weight to poor quality samples could be useful. Increasing the domain of training data results in better performance for some tasks. Selecting best combination of datasets based on frozen BERT with intent to be trained with unfrozen BERT would not work. 

📜 See final report in the repo above


## Datasets:

❗❗❗ All datasets except the first one contain offensive jokes, be careful

😂 _Humicroedit and FunLines datasets_

Nabil Hossain, John Krumm and Michael Gamon. "President Vows to Cut ~~Taxes~~ Hair": Dataset and Analysis of Creative Text Editing for Humorous Headlines. 2019. In NAACL. 

FunLines - Nabil Hossain, John Krumm Tanvir Sajed and Henry Kautz. "Stimulating Creativity with FunLines: A Case Study of Humor Generation in Headlines. arXiv preprint (2020). 

__Description__: Both of the datasets have an original news title and “microedited” one, where one word was replaced to make is funny. The readers then assigned a score from 0 to 3; each title has multiple people reviewing it. 

__Download from__: https://www.cs.rochester.edu/u/nhossain/humicroedit.html


😂 _CrowdTruth Short-Text-Corpus-For-Humor-Detection_

__Description__: Scraped from twitter, contains posts from “funny” accounts, as well as Reuters headlines, English proverbs and Wiki sentences. This results in approximately 22K funny items, and 21K of neutral posts.

__Download from__: https://github.com/CrowdTruth/Short-Text-Corpus-For-Humor-Detection


😂 _Kaggle, Short Jokes_ 

__Download from__: https://www.kaggle.com/abhinavmoudgil95/short-jokes


😂_Puns, reddit full and short jokes_

__Download from__: https://github.com/orionw/RedditHumorDetection


🤔 For additional balancing: _A Million News Headlines_

Use most recent 200K news titles
https://www.kaggle.com/therohk/million-headlines

# Contents of notebook datasets

_BERT.ipynb_ contains a simlple working fine-tuning example

_Dataset analysis and preparation.ipynb_ contains dataset description and data mixing explanations

_dataset comparison and assemble.ipynb_ mixing data

_Humicroedit.ipynb_ a closer look at Humicroedit dataset since it is an interesting one





# Additional

Followed this tutorial quite a bit https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
