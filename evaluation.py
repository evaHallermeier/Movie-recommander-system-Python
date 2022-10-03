# Eva Hallermeier 337914121

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math




#check precision for single user which we find recommandation before
def precision_10(test_set, cf, is_user_based = True):
    val = 0
    nbOfrecommandations= 10
    test_users = test_set['userId']
    test_users = test_users.unique()  #list of users id in testset
    nb_ofDifferent_Users = test_set['userId'].value_counts().size
    hightratings = test_set.loc[test_set['rating'] >= 4.0]  # get highratings from testset
    test_users = test_users.tolist()
    for userId in test_users:
        testBestMoviesForUser = ((hightratings.loc[hightratings['userId'] == userId]['movieId'] )) #get movies id from hirghrating for specific user
        testBestMoviesForUser = testBestMoviesForUser.tolist()
        systemTop10 = cf.predict_moviesForEvaluation(userId, nbOfrecommandations, is_user_based) # get id of top k movies from cf for specific user
        val = val + ((len(set(systemTop10).intersection(testBestMoviesForUser))) / nbOfrecommandations) #wants to know nb of hits in this top k

    #divide val by nb users in test
    val = val / nb_ofDifferent_Users  #average precision for al users
    print("Precision_k: " + str(val))

# this metrics check position of each recommandation and check if it s a favorite movie by the user in test
def ARHA(test_set, cf, is_user_based = True):
    val = 0
    nbOfRecommandations = 10
    test_users = test_set['userId']
    test_users = test_users.unique()

    nb_ofDifferent_Users = test_set['userId'].value_counts().size
    hightratings = test_set.loc[test_set['rating'] >= 4.0] # get hirghratings of all users from test
    test_users = test_users.tolist()
    for userId in test_users:
        testBestMoviesForUser = ((hightratings.loc[hightratings['userId'] == userId]['movieId'])) # get highratings for specific user
        testBestMoviesForUser = testBestMoviesForUser.tolist()
        systemTop10 = cf.predict_moviesForEvaluation(userId, nbOfRecommandations, is_user_based) # get id of recommandations movies find by predict function in cf
        for i in range(nbOfRecommandations): # to over each recommandation stored in systemTop10
            n = 1 / (i + 1) if systemTop10[i] in testBestMoviesForUser else 0
            val = val + n

    val = val / nb_ofDifferent_Users    #divide val by nb users in test
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based = True):
    pred=[]
    act =[]
    nb_ofDifferent_Users = test_set['userId'].value_counts().size
    val = 0
    actualRatings = test_set.pivot_table("rating", index=["userId"], columns="movieId")
    actualRatings = actualRatings.to_numpy()
    nb_users_id = len(cf.getUSERS_id()) - 1

    if is_user_based:
        predictedRatings = cf.get_user_based_matrix()

    else:
        predictedRatings = cf.get_item_based_matrix()

    predictedRatings = np.delete(predictedRatings, nb_users_id ,0) # there is an additional user in prediction so we delete him for computation
    for i in range((nb_ofDifferent_Users)):
        p = predictedRatings[i] # predictions for specific user
        a = actualRatings[i] # ratings get in testset for specific user
        indexnanvalues = np.argwhere(np.isnan(a)).tolist()
        a = a[~np.isnan(a)] #remove nan
        act.extend(a) # add actual ratings of the user in list of global actual valules (act)
        p = np.delete(p, indexnanvalues) # remove movies that are not in testset (by user)
        pred.extend(p)  # add prediction of specific user (p) in list of prediction (pred)
    val = math.sqrt(mean_squared_error(act, pred)) # formula for RSME
    print("RMSE: " + str(val))