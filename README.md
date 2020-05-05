# humor_project
Scoring how funny phrases are, Princeton COS598C final project


# Datasets used:

_HashtagWars dataset_ : 

"Semeval-2017 task 6:# hashtagwars: Learning a sense of humor." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017. 

__Description__: Scraped from twitter, and uses a Comedy Central TV game-show. Twitter users try to come up with funny punchlines following the hashtag topic. Total of 12734 tweets for 112 hashtags.

Downloaded from: http://alt.qcri.org/semeval2017/task6/index.php?id=data-and-tools

Data structure: single directory w/ files corresponding to a single hashtag. Each line has format tweet_id tweet_text tweet_label, with labels: 
- 0: not in the top 10
- 1: top 10, but not winning (90 out of 101 files have all 9, the rest have 8)
- 2: winning tweet (one tweet per file)


