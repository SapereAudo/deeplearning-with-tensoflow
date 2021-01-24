import numpy as np
import pandas as pd
from itertools import combinations
from random import randint

# 负样本与正样本比例
rate = 3
context_length = 7

def sample(group,context_length):
    f = open("word2VecTrainData.txt","w+")
    for uid,movieLs in group.items():
        i,j=0,context_length
        total = 0
        while j<len(movieLs):
            sample = context_sample(movieLs[i:j])
            i+=1
            j+=1
            total += len(sample)
            for s in sample:
                f.write("%s\n"%(s))
        print("user {}'s sample has Done,with {} samples".format(uid,total))
    f.close()


def context_sample(sequence):
    # 序列长度取奇数
    positive = []
    center = sequence[len(sequence) // 2]
    for i in sequence:
        if i == center: continue
        positive.append("{}|{}|{}".format(center, i, 1))
        t = randint(movieIds[0], movieIds[-1])

    negative = []
    n = len(sequence) - 1

    while len(negative) < n * rate:
        t = randint(movieIds[0], movieIds[-1])
        if t in sequence: continue
        negative.append("{}|{}|{}".format(center, t, 0))
    return positive + negative

if __name__ == '__main__':
    ratings = pd.read_csv("../ml-1m/ratings.dat",sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
    )
    movies = pd.read_csv(
        "../ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"]
    )
    ratings_group = dict(ratings.sort_values(by=["unix_timestamp"]).groupby("user_id").movie_id.apply(list))
    movieIds = movies.movie_id.unique()

    sample(ratings_group, context_length)

    data = pd.read_csv("./word2VecTrainData.txt", sep="|", names=["center", "context", "label"])
    random_selection = np.random.rand(len(data.index)) <= 0.85
    train_data = data[random_selection]
    test_data = data[~random_selection]
    train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
    test_data.to_csv("test_data.csv", index=False, sep="|", header=False)