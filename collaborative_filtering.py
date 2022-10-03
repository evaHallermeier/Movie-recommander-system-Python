# Eva Hallermeier 337914121

import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances

class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []     #matrix of prediction for CF user based algorithm
        self.item_based_matrix = []     #matrix of prediction for CF item based algorithm
        self.users_id =[]               #array that map all user between his id and index in the matrix of prediction
        self.movies_id =[]              #array that map all movie between his id and index in the matrix of prediction

    def getUSERS_id(self):
         return self.users_id

    def get_user_based_matrix(self):
        return self.user_based_matrix

    def get_item_based_matrix(self):
        return self.item_based_matrix

    def create_fake_user(self,rating):
        df = rating
        fake_user_id = 283238
        #user that like children content and not crime or action
        df1 = {'userId': fake_user_id, 'movieId': 1, 'rating': 4.5 } #toy story
        df2 = {'userId': fake_user_id, 'movieId': 837, 'rating': 4.5}  # matilda
        df3 = {'userId': fake_user_id, 'movieId': 3034, 'rating': 5.0}  # robin hood children
        df4 = {'userId': fake_user_id, 'movieId': 6957, 'rating':1.5}  # bad santa -comedy crime
        df5 = {'userId': fake_user_id, 'movieId': 86880, 'rating':3.0}  # pirates of the caribbean
        df6 = {'userId': fake_user_id, 'movieId': 46972, 'rating':5.0}  # night at the museum - comedy fantasy
        df7 = {'userId': fake_user_id, 'movieId': 54001, 'rating':2.0}  # harry potter
        df8 = {'userId': fake_user_id, 'movieId': 52281, 'rating':1.0}  # grindhouse horror

        #add those rows in ratings
        df = df.append(df1, ignore_index=True)
        df = df.append(df2, ignore_index=True)
        df = df.append(df3, ignore_index=True)
        df = df.append(df4, ignore_index=True)
        df = df.append(df5, ignore_index=True)
        df = df.append(df6, ignore_index=True)
        df = df.append(df7, ignore_index=True)
        df = df.append(df8, ignore_index=True)

        return df


    def create_user_based_matrix(self, data):
        ratings = data[0]

        # for adding fake user - in question 5
        #ratings = self.create_fake_user(ratings)
        #####################

        movies = data[1]
        self.movies_data = movies
        pivotTable = ratings.pivot_table("rating", index=["userId"], columns="movieId")
        self.ratingsPerUsers = pivotTable
        self.movies_id = np.array(pivotTable.columns)
        self.users_id = np.array(pivotTable.index)
        ratings_np = pivotTable.to_numpy()
        mean_user_rating = pivotTable.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (ratings_np - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        pd.DataFrame(user_similarity)
        pd.DataFrame(user_similarity.dot(ratings_diff))
        self.user_based_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    def create_item_based_matrix(self, data):
        ratings = data[0]
        self.ratings = ratings
        movies = data[1]
        self.movies_data = movies
        pivotTable = ratings.pivot_table("rating", index=["userId"], columns="movieId")
        self.ratingsPerUsers = pivotTable
        self.movies_id = np.array(pivotTable.columns)
        self.users_id = np.array(pivotTable.index)
        ratings_np = pivotTable.to_numpy()
        mean_user_rating = pivotTable.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (ratings_np - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        ratingItem = ratings_diff
        ratingItem[np.isnan(ratingItem)] = 0
        item_similarity = 1 - pairwise_distances(ratingItem.T, metric='cosine')
        pd.DataFrame(item_similarity)
        self.item_based_matrix = mean_user_rating + ratingItem.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])


    def predict_moviesForEvaluation(self, user_id, k, is_user_based = True):  #return movie id and not titles
        user_id_inTable = np.where(self.users_id == int(user_id))  # translate for table
        user_id_inTable = user_id_inTable[0].item(0)
        ratings_of_user = self.ratingsPerUsers.iloc[user_id_inTable] #index
        # are movie id and then column of rating for each movie for the user asked
        movieIndex_notRated = np.where(ratings_of_user.isna())[0]

        if is_user_based:
            user_prediction = self.user_based_matrix[user_id_inTable]
        else:
            user_prediction = self.item_based_matrix[user_id_inTable]

        movies_recommandated_byindex = movieIndex_notRated[(-user_prediction[movieIndex_notRated]).argsort()[:k]]
        # get k best movies not rated (we want sort but desending order)
        movies_recommandated_byID = self.movies_id[movies_recommandated_byindex]
        return movies_recommandated_byID

    def predict_movies(self, user_id, k, is_user_based = True):
        results=[]
        movies_recommandated_byID = self.predict_moviesForEvaluation(user_id, k, is_user_based)
        for i in range(k): #for each recommandation movie
            movieData = self.movies_data[self.movies_data['movieId'] == movies_recommandated_byID[i]]['title']
            # get title of movie from id
            movie = movieData.values[0]
            results.append(movie)
        return results