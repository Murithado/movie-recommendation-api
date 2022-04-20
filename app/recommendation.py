import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data():
    '''
    This retrieves the preprocessed data from the zipped csv file
    '''
    movie_data = pd.read_csv('app/dataset/movie_data.csv.zip')
    movie_data['original_title'] = movie_data['original_title'].str.lower()
    return movie_data

def combine_data(data):
    '''
    The function takes data from the get_data function and the cast and the genres are combined together to be made into a bag of words
    '''
    data_combined = data.drop(columns=['movie_id', 'plot', 'original_title'])
    data_combined['combine'] = data_combined[data_combined.columns[:3]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

    data_combined = data_combined.drop(columns=['cast', 'genres', 'release_date'])
    return data_combined

def transform_data(movie_data, combined_data):
    '''
    Takes the unmodified data from get_data() and the combined data from combine_data() and returns the cosine similarity matrix
    '''
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(combined_data['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_data['plot'])

    combined_matrix = sp.hstack([count_matrix, tfidf_matrix], format='csr')

    cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

    return cosine_sim

def recommend_movies(title, movie_data, combined_data, transformed_data):
    '''
    It takes in a movie title and returns information or 20 closely related movies based off the cosine similarity score
    '''
    index = movie_data[movie_data['original_title'] == title].index[0]

    cossim_scores = list(enumerate(transformed_data[index]))
    cossim_scores = sorted(cossim_scores, key=lambda x: x[1], reverse=True)
    cossim_scores = cossim_scores[1:21]

    movie_indices = [c[0] for c in cossim_scores]
    
    recommended_data = pd.DataFrame(columns=['movie_id', 'name', 'genre'])
    recommended_data['movie_id'] = movie_data.loc[movie_indices, 'movie_id']
    recommended_data['name'] = movie_data.loc[movie_indices, 'original_title']
    recommended_data['genre'] = movie_data.loc[movie_indices, 'genres']

    return recommended_data

def results(movie_name):
    '''
    It takes in the name of the movie and returns the records of the 20 related movies
    '''
    movie_name = movie_name.lower()

    movie_dframe = get_data()

    combined_dframe = combine_data(movie_dframe)

    transformed_array = transform_data(movie_dframe, combined_dframe)

    if movie_name not in movie_dframe['original_title'].unique():
        return 'Movie is not in Database'

    else:
        recommendations = recommend_movies(movie_name, movie_dframe, combined_dframe, transformed_array)
        return recommendations.to_dict('records')






    