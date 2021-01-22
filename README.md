# Humor Project

This is a repository for Princeton COS598C final project, Spring 2020, taught by Prof. Danqi Chen. 

👉 In short, the project explored the performance of frozen/unfrozen pre-trained `BERT-base-uncased` models on the joke recognition, and the effect dataset formation has on the performance. 

📜 See final report in the repo above


## Datasets:

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


🤔 For balancing:

_A Million News Headlines_

Use most recent 200K news titles
https://www.kaggle.com/therohk/million-headlines

💭 Other possible sources (not used in this project)

_HashtagWars dataset_  

"Semeval-2017 task 6:# hashtagwars: Learning a sense of humor." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017. 

__Description__: Scraped from twitter, and uses a Comedy Central TV game-show. Twitter users try to come up with funny punchlines following the hashtag topic. Total of 12734 tweets for 112 hashtags.

__Download from__: http://alt.qcri.org/semeval2017/task6/index.php?id=data-and-tools

__Data structure__: single directory w/ files corresponding to a single hashtag. Each line has format tweet_id tweet_text tweet_label, with labels: 
- 0: not in the top 10
- 1: top 10, but not winning (90 out of 101 files have all 9, the rest have 8)
- 2: winning tweet (one tweet per file)

MadLibs humor 
https://www.microsoft.com/en-us/download/details.aspx?id=55593

Twitter random tweets
https://archive.org/search.php?query=collection%3Atwitterstream&sort=-publicdate

One week of global news
https://www.kaggle.com/therohk/global-news-week
