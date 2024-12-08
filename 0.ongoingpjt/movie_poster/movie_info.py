import pandas as pd
import requests
import streamlit as st
from apikey import omdb_apikey, openai_apikey, slack_webhook_url

OMDB_API_URL = f'http://www.omdbapi.com/'
OMDB_API_KEY = omdb_apikey
OPENAI_API_KEY = openai_apikey
SLACK_WEBHOOK_URL = slack_webhook_url

def get_movie_info(title):
    try:
        search_params = {
            'apikey': OMDB_API_KEY,
        't': title
        }
        response = requests.get(OMDB_API_URL, params=search_params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['Response'] == 'True':
            return data
        else:
            print(f"Movie not found: {title}", data.get('Error'))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def create_poster_prompt(movie_info):
    title = movie_info.get('Title')
    genre = movie_info.get('Genre')
    plot = movie_info.get('Plot')
    rating = movie_info.get('imdbRating')
    top_rating = movie_info.get('Ratings')[0]['Value']

    prompt = (
        f"Create a movie poster for '{title}'\n\n"
        f"Plot: {plot}\n"
        f"Genre: {genre}\n"
        f"Rating: {rating}/10\n\n"
        f"Style: Create a visually stunning poster that captures the essence of this {genre.lower()} "
        f"film. Incorporate key visual elements that evoke the movie's themes and atmosphere."
    )
    return prompt

def main():
    st.title('Movie Poster Generator')
    st.write("Enter a movie title to generate a poster prompt.")

    movie_title = st.text_input('Enter a movie title')

    if st.button('Get Movie Info'):
        if movie_title:
            movie_info = get_movie_info(movie_title)
            if movie_info:
                st.subheader('Movie Details')
                st.write(f"Title: {movie_info['Title']}")
                st.write(f"Genre: {movie_info['Genre']}")
                st.write(f"Plot: {movie_info['Plot']}")
                st.write(f"Rating: {', '.join([rating['Value'] for rating in movie_info['Ratings']])}")
                
                poster_prompt = create_poster_prompt(movie_info)
                st.subheader('Poster Prompt')
                st.code(poster_prompt)
            else:
                st.error('Failed to fetch movie details. Please try again.')
        else:
            st.warning('Please enter a movie title')

if __name__ == '__main__':
    main()