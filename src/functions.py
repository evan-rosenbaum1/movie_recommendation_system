import pandas as pd
from surprise import accuracy
from surprise.model_selection import GridSearchCV as surprise_GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def surprise_tune_and_evaluate(model, param_grid, data_train, test_data, model_name, collab_models_log):
    # Set up GridSearchCV
    gs = surprise_GridSearchCV(model, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
    gs.fit(data_train)
    
    # Get the best model and parameters
    best_model = gs.best_estimator['rmse']

    # Train on the entire trainset
    trainset = data_train.build_full_trainset()
    best_model.fit(trainset)
    
    # Calculate the RMSE score for the trainset
    print('Trainset Results')
    predictions_trainset = best_model.test(trainset.build_testset())
    rmse_train = round(accuracy.rmse(predictions_trainset), 5)

    # Fit the best model on the holdout test
    print('Testset Results')
    testset = data_train.construct_testset(test_data) # Converts data_train from train to test data
    predictions_testset = best_model.test(testset)
    rmse_test = round(accuracy.rmse(predictions_testset), 5)

    # Log RMSE
    collab_models_log[model_name] = {
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test
    }
    
    return best_model

def boosting_tune_and_evaluate(model, param_grid, X_train, X_test, y_train, y_test, model_name, collab_models_log):
    # Set up GridSearchCV
    gs = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = gs.best_estimator_

    # Evaluate the model on the training set
    y_train_pred = best_model.predict(X_train)
    rmse_train = round(mean_squared_error(y_train, y_train_pred, squared=False), 5)
    print(f'{model_name} Model Train RMSE: {rmse_train}')

    # Evaluate the model on the test set
    y_test_pred = best_model.predict(X_test)
    rmse_test = round(mean_squared_error(y_test, y_test_pred, squared=False), 5)
    print(f'{model_name} Model Test RMSE: {rmse_test}')

    # Log RMSE
    collab_models_log[model_name] = {
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test
    }

    return best_model

def get_top_rated_movies(user_id, collab_filter_df, movies_df, n=10):
    user_ratings = collab_filter_df[collab_filter_df['user_id'] == user_id]
    top_rated_movies = user_ratings.sort_values(by='rating', ascending=False).head(n)
    top_rated_movies = pd.merge(top_rated_movies, movies_df, on='movie_id')
    return top_rated_movies[['movie_id','title', 'rating']]

def get_top_n_recommendations_feat_gbm(user_id, predictions_df, collab_filter_df, movies_df, n=10):
    # Filter for the specific user
    user_predictions = predictions_df[predictions_df['user_id'] == user_id]
    
    # Exclude movies already rated by the user
    rated_movies = collab_filter_df[collab_filter_df['user_id'] == user_id]['movie_id']
    user_predictions = user_predictions[~user_predictions['movie_id'].isin(rated_movies)]
    
    # Sort by the final predicted rating and get the top N
    top_n_recommendations = user_predictions.sort_values(by='final_feat_gbm_est', ascending=False).head(n)
    
    # Merge with movies_df to get movie titles
    top_n_recommendations = pd.merge(top_n_recommendations, movies_df, on='movie_id')
    
    return top_n_recommendations[['title_x', 'movie_id', 'final_feat_gbm_est']]

def get_top_n_recommendations_gbm(user_id, predictions_df, collab_filter_df, movies_df, n=10):
    # Filter for the specific user
    user_predictions = predictions_df[predictions_df['user_id'] == user_id]
    
    # Exclude movies already rated by the user
    rated_movies = collab_filter_df[collab_filter_df['user_id'] == user_id]['movie_id']
    user_predictions = user_predictions[~user_predictions['movie_id'].isin(rated_movies)]
    
    # Sort by the final predicted rating and get the top N
    top_n_recommendations = user_predictions.sort_values(by='final_gbm_est', ascending=False).head(n)
    
    # Merge with movies_df to get movie titles
    top_n_recommendations = pd.merge(top_n_recommendations, movies_df, on='movie_id')
    
    return top_n_recommendations[['title_x', 'movie_id', 'final_gbm_est']]


def get_top_n_recommendations_svd(user_id, predictions_df, collab_filter_df, movies_df, n=10):
    # Filter for the specific user
    user_predictions = predictions_df[predictions_df['user_id'] == user_id]
    
    # Exclude movies already rated by the user
    rated_movies = collab_filter_df[collab_filter_df['user_id'] == user_id]['movie_id']
    user_predictions = user_predictions[~user_predictions['movie_id'].isin(rated_movies)]
    
    # Sort by the final predicted rating and get the top N
    top_n_recommendations = user_predictions.sort_values(by='est', ascending=False).head(n)
    
    # Merge with movies_df to get movie titles
    top_n_recommendations = pd.merge(top_n_recommendations, movies_df, on='movie_id')
    
    return top_n_recommendations[['title_x', 'movie_id', 'est']]

def compute_similarity_matrix(predictions_df):
    # Select the movie features (genre columns)
    movie_features_df = predictions_df.drop(columns=['user_id', 'true_r', 'est', 'details', 'svd_residual', 
                                                     'title', 'gbm_est', 'final_gbm_est', 'feat_gbm_est'])
    
    feature_columns = movie_features_df.drop(['movie_id'], axis=1).columns

    # Normalize the features
    scaler = StandardScaler()
    movie_features_df[feature_columns] = scaler.fit_transform(movie_features_df[feature_columns])

    # Group by 'movie_id' and aggregate mean for each feature to avoid multiple rows for the same movie
    movie_features = movie_features_df.groupby('movie_id')[feature_columns].mean().reset_index()

    # Extract feature matrix for similarity calculation
    feature_matrix = movie_features[feature_columns].values

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)

    # Convert to DataFrame for easier manipulation
    similarity_df = pd.DataFrame(similarity_matrix, index=movie_features['movie_id'], columns=movie_features['movie_id'])
    
    return similarity_df

def get_similar_movies(movie_id, similarity_df, n=3):
    # Get similarity scores for the given movie_id
    similarity_scores = similarity_df.loc[movie_id]

    # Get indices of the top N similar movies (excluding the movie itself)
    similar_movie_ids = similarity_scores.nlargest(n+1).iloc[1:].index.values

    return similar_movie_ids

def get_recommendations_based_on_similarity(user_id, predictions_df, collab_filter_df, movies_df, similarity_df, top_n=10, n_similar=3):
    # Get the user's top N recommendations
    top_n_recommendations = get_top_n_recommendations_feat_gbm(user_id, predictions_df, collab_filter_df, movies_df, n=top_n)
    
    final_recommendations = []

    for _, row in top_n_recommendations.iterrows():
        top_movie_id = row['movie_id']
        top_movie_title = row['title_x']
        top_movie_rating = row['final_feat_gbm_est']
        
        # Get similar movies
        similar_movie_ids = get_similar_movies(top_movie_id, similarity_df, n=n_similar)
        
        for similar_movie_id in similar_movie_ids:
            similar_movie_title = movies_df[movies_df['movie_id'] == similar_movie_id]['title'].values[0]
            similarity_score = similarity_df.loc[top_movie_id, similar_movie_id]
            
            # Get the predicted rating for the similar movie
            similar_movie_rating = predictions_df[(predictions_df['movie_id'] == similar_movie_id) & 
                                                  (predictions_df['user_id'] == user_id)]['final_feat_gbm_est'].values[0]
            
            final_recommendations.append({
                'Top Rated Movie': top_movie_title,
                'Recommended Movie': similar_movie_title,
                'Similarity Score': similarity_score,
                'Predicted Rating': similar_movie_rating
            })
    
    # Create DataFrame from the recommendations list
    recommendations_df = pd.DataFrame(final_recommendations)
    
    # Maintain the order of top-rated movies and sort by predicted rating within each group
    recommendations_df = recommendations_df.sort_values(by=['Top Rated Movie', 'Predicted Rating'], ascending=[True, False]).reset_index(drop=True)
    
    return recommendations_df