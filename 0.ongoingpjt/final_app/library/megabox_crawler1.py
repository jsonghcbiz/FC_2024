from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import os
import time

MEGABOX_SEARCH_URL = "https://www.megabox.co.kr/movie?searchText="

class MegaboxTitleCrawler:
    def __init__(self):
        pass
        
    def fetch_movie_titles(self, search_query):
        # Example of debugging output
        print(f"Searching for: {search_query}")
        
        driver = webdriver.Chrome()
        try:
            search_url = f"https://www.megabox.co.kr/movie?searchText={search_query}"
            driver.get(search_url)

            wait = WebDriverWait(driver, 5)
            movie_list_container = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".movie-list ol.list#movieList"))
            )


            movie_elements = movie_list_container.find_elements(By.CSS_SELECTOR, ".movie-list-info")
            movie_title_list = []
            movie_info_list = []
            
            print(f"Found {len(movie_elements)} movie elements.")  # Debugging line

            for movie in movie_elements:
                try:
                    # Debugging: Print the HTML of the movie element
                    print(movie.get_attribute('outerHTML'))  # Print the HTML of the current movie element

                    # Extract movie title from the alt attribute of the img tag
                    movie_title = movie.find_element(By.CSS_SELECTOR, "img.poster").get_attribute("alt")
                    movie_title_list.append(movie_title)

                    # Add any additional movie info (like summary or score if needed)
                    movie_summary = movie.find_element(By.CSS_SELECTOR, "div.summary").text.strip()
                    movie_score = movie.find_element(By.CSS_SELECTOR, "div.my-score .number").text.strip()

                    movie_info_list.append({
                        "title": movie_title,
                        "summary": movie_summary,
                        "rating": movie_score,
                    })
                except Exception as e:
                    print(f"Error extracting movie info: {e}")
                    continue

            print(f"영화 제목: {movie_title_list}")  # Debugging line
            print("Fetched Titles:", movie_title_list)
            return movie_title_list, movie_info_list

        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {str(e)}")
            return [], []
    
class MegaboxDetailCrawler:
    def __init__(self):
        self.driver = None
    
    def navigate_to_movie_detail_page(self, movie_title):
        self.driver = webdriver.Chrome()
        try:
            self.driver.get(MEGABOX_SEARCH_URL + movie_title)
            
            wait = WebDriverWait(self.driver, 5)
            print("영화 상세 페이지로 이동중...")
            detail_button = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "movie-list-info"))
            )
            detail_button.click()
            return True

        except Exception as e:
            print(f"영화 상세 페이지로 이동 중 오류가 발생했습니다: {e}")
            return False
    
    def fetch_plot_information(self):
        wait = WebDriverWait(self.driver, 5)
        print("줄거리 정보 크롤링중...")
        try:
            time.sleep(1)
            selectors = [
                "div.movie-summary.infoContent.on div.txt",
                "div.movie-summary div.txt",
                "div.txt",
                ".movie-info .txt",
                "#info div.txt"
            ]
            
            for selector in selectors:
                try:
                    print(f"Trying selector: {selector}")
                    plot = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    ).text
                    if plot:
                        print(f"Found plot using selector: {selector}")
                        break
                except Exception as e:
                    print(f"Selector {selector} not found: {e}")
                    continue
            print(f"Final plot: {plot}")
            return plot
        except Exception as e:
            print(f"Error fetching plot: {e}")
            return "정보 없음"

    def fetch_director_and_cast(self):
        wait = WebDriverWait(self.driver, 5)
        print("감독/출연진 정보 크롤링중...")
        try:
            # Extract director information
            director_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.movie-info.infoContent .line p"))
            )
            directors = []
            for element in director_elements:
                if "감독" in element.text:
                    directors.append(element.text.split(":")[1].strip())

            # Extract cast information
            cast_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.movie-info.infoContent p"))
            )
            casts = []
            for element in cast_elements:
                if "출연진" in element.text:
                    casts = element.text.split(":")[1].strip().split(", ")  # Split by comma for multiple cast members
                    break  # Stop after finding the cast information

            return directors, casts if casts else ["정보 없음"]  # Handle missing cast information
        except Exception as e:
            print(f"감독/출연진 정보 크롤링 중 오류 발생: {e}")
            return [], []


    def fetch_reviews(self, review_limit=100):
        wait = WebDriverWait(self.driver, 5)
        print("리뷰 정보 크롤링중...")
        try:
            # Wait for the movie detail page to load completely
            time.sleep(1)

            # Find and click the review tab using JavaScript
            try:
                review_tab = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".tab-list.fixed ul li:nth-child(2) a"))
                )
                self.driver.execute_script("arguments[0].click();", review_tab)
                print("Clicked review tab")
                time.sleep(3)  # Wait for reviews to load
            except Exception as e:
                print(f"Error clicking review tab: {e}")
                return pd.DataFrame(columns=["rating", "review"])

            review_list = []
            page_no = 1
            
            while len(review_list) < review_limit:
                try:
                    # Wait for reviews to be visible
                    reviews_container = wait.until(
                        EC.presence_of_element_located((By.CLASS_NAME, "story-box"))
                    )
                    
                    # Find all review items
                    review_items = self.driver.find_elements(By.CSS_SELECTOR, ".story-box .story-wrap.review")
                    print(f"Found {len(review_items)} reviews on page {page_no}")

                    if not review_items:
                        print("No reviews found on this page")
                        break

                    for item in review_items:
                        try:
                            # Extract rating
                            rating_element = item.find_element(By.CSS_SELECTOR, ".story-point span")
                            rating = rating_element.text.strip()
                            
                            # Extract review text
                            review_element = item.find_element(By.CSS_SELECTOR, ".story-txt")
                            review_text = review_element.text.strip()
                            
                            print(f"Extracted review - Rating: {rating}, Text: {review_text[:50]}...")  # Debug print
                            
                            review_list.append({
                                "rating": rating,
                                "review": review_text
                            })

                            if len(review_list) >= review_limit:
                                break
                        except Exception as e:
                            print(f"Error extracting review details: {e}")
                            continue

                    # Try to go to next page if we need more reviews
                    if len(review_list) < review_limit:
                        try:
                            # Wait for pagination element to be present
                            pagination = wait.until(
                                EC.presence_of_element_located((By.CLASS_NAME, "pagination"))
                            )
                            
                            # Find the current page number from the active element
                            current_page = pagination.find_element(By.CSS_SELECTOR, "strong.active").text
                            next_page_num = int(current_page) + 1
                            
                            # Try to find the next page link by pagenum attribute
                            next_page_link = pagination.find_element(
                                By.CSS_SELECTOR, 
                                f"a[pagenum='{next_page_num}']"
                            )
                            
                            if next_page_link:
                                # Click the next page link using JavaScript
                                self.driver.execute_script("arguments[0].click();", next_page_link)
                                print(f"Navigating to page {next_page_num}")
                                time.sleep(2)  # Wait for the new page to load
                                page_no += 1
                            else:
                                print("No more pages available")
                                break

                        except NoSuchElementException as e:
                            print(f"Pagination error: {e}")
                            break
                        except Exception as e:
                            print(f"Error during pagination: {e}")
                            break

                except Exception as e:
                    print(f"Error on page {page_no}: {e}")
                    print(f"Detailed error: {str(e)}")
                    break

            print(f"Total reviews collected: {len(review_list)}")
            return pd.DataFrame(review_list[:review_limit])

        except Exception as e:
            print(f"리뷰 정보 크롤링 중 오류 발생: {e}")
            print(f"Detailed error: {str(e)}")
            return pd.DataFrame(columns=["rating", "review"])
    
    def crawl_movie_details(self, movie_title, review_limit=100):
        try:
            self.navigate_to_movie_detail_page(movie_title)
            plot = self.fetch_plot_information()
            directors, casts = self.fetch_director_and_cast()
            reviews = self.fetch_reviews(review_limit)

            directors_str = ', '.join(directors)
            casts_str = ', '.join(casts) if casts else '정보 없음'  # Handle missing cast information

            movie_data = {
                "Title": movie_title,
                "Plot": plot,
                "Directors": directors_str,
                "Casts": casts_str,  # Ensure this is populated correctly
                "Reviews": reviews
            }

            return movie_data

        except Exception as e:
            print(f"영화 상세 정보 크롤링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            if self.driver:
                self.driver.quit() 