import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import tkinter as tk
from tkinter import ttk, messagebox

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

ps = PorterStemmer()

def convert(obj):
    L = []
    for i in ast.literal_eval(obj): 
        L.append(i['name'])
    return L

def convert3(obj):
    l = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:  
            l.append(i['name'])
            count += 1
        else:
            break
    return l

def fetchDirector(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetchDirector)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in new_df['title'].values:
        return ["Movie not found!"]
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    
    return recommended_movies

def on_recommend():
    movie = movie_entry.get()
    if movie.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a movie title")
        return
    recommendations = recommend(movie)
    if recommendations == ["Movie not found!"]:
        messagebox.showerror("Error", "Movie not found!")
    else:
        recommendations_text = "\n".join(recommendations)
        recommendations_label.config(text=recommendations_text)

app = tk.Tk()
app.title("Movie Recommender System")
app.geometry("600x500")
app.resizable(False, False)

style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 12), padding=10)
style.configure("TEntry", font=("Arial", 12), padding=10)

app.configure(bg="#f5f5f5")

title_label = tk.Label(app, text="Movie Recommender System", font=("Arial", 20, "bold"), bg="#f5f5f5", fg="#333333")
title_label.pack(pady=20)

movie_frame = tk.Frame(app, bg="#f5f5f5")
movie_frame.pack(pady=10)

movie_label = tk.Label(movie_frame, text="Enter a movie title:", font=("Arial", 14), bg="#f5f5f5", fg="#333333")
movie_label.grid(row=0, column=0, padx=10)

movie_entry = ttk.Entry(movie_frame, width=40)
movie_entry.grid(row=0, column=1, padx=10)

recommend_button = ttk.Button(app, text="Recommend", command=on_recommend)
recommend_button.pack(pady=20)

recommendations_label = tk.Label(app, text="", font=("Arial", 12), bg="#f5f5f5", fg="#333333", justify="left")
recommendations_label.pack(pady=20)

app.mainloop()