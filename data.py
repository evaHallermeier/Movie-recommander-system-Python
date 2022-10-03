

# import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def watch_data_info(data): #given function
        for d in data:
                # This function returns the first 5 rows for the object based on position.
                # It is useful for quickly testing if your object has the right type of data in it.
                print(d.head())

                # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
                print(d.info())

                # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
                print(d.describe(include='all').transpose())

# print all data asked in questions of the exercice
def print_data(data):
        d = data[0]

        nb_ofDifferent_Users = d['userId'].value_counts().size
        nb_ofDifferent_Movies = d['movieId'].value_counts().size
        print("There are {} different users that gave ratings on movies.".format(nb_ofDifferent_Users))

        print("There are {} different movies that received ratings.".format(nb_ofDifferent_Movies))
        i = d.index
        nb_ofRatings = len(i)

        print("There are in total {} ratings.".format(nb_ofRatings))

        nbRatingsPerMovies = d['movieId'].value_counts()
        MaxNB_of_ratingsForMovie = nbRatingsPerMovies.head(1).item()
        MinNB_of_ratingsForMovie = nbRatingsPerMovies.tail(1).item()
        print("The minimum number of votes a movie received was {} and the maximum was {}.".format(MinNB_of_ratingsForMovie,
                                                                          MaxNB_of_ratingsForMovie))

        nbRatingsPerUser = d['userId'].value_counts()
        MaxNB_of_ratingsForUser = nbRatingsPerUser.head(1).item()
        MinNB_of_ratingsForUser = nbRatingsPerUser.tail(1).item()
        print("The minimal amount of votes given by  a user is {} and maximum is {} votes.".format(MinNB_of_ratingsForUser,
                                                                                                MaxNB_of_ratingsForUser))
# create and show plot of distribution of ratings values
def plot_data(data, plot = True):
        d = data[0]
        occurenceofgrades = d['rating'].value_counts()
        occurenceofgrades = occurenceofgrades.sort_index(ascending=True)
        y = occurenceofgrades.values
        x = occurenceofgrades.index
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.bar(x, y, color='blue',width=0.45)
        plt.xlabel("rating value", horizontalalignment='center', fontweight='bold')
        plt.ylabel("frequency")
        plt.title("Distribution of rating value")
        if(plot):
                plt.show()
