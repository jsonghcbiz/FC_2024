from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import os
import time


CGV_SEARCH_URL = "http://www.cgv.co.kr/search/?query="

class CGVTitleCrawler:
    def __init__(self):
        pass
        
    def fetch_movie_titles(self, search_text):
        driver = webdriver.Chrome()      ##############################
        try:
            search_url = f"{CGV_SEARCH_URL}{search_text}"
            driver.get(search_url)

            wait = WebDriverWait(driver, 10)
            movie_list_container = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "searchingMovieResult_list"))
            )

            # Find all strong elements with class 'searchingMovieName'
            movie_elements = movie_list_container.find_elements(By.CSS_SELECTOR, "strong.searchingMovieName")
            
            movie_title_list = []
            movie_info_list = []  # List to store both title and rating
            for element in movie_elements:
                # Get the text content and split by newline
                full_text = element.text.strip()
                parts = full_text.split('\n')
                title = parts[0].strip()
                
                # Try to get age rating if available
                # age_rating = parts[1].strip() if len(parts) > 1 else "연령정보 없음"
                
                movie_title_list.append(title)
                movie_info_list.append({
                    "title": title
                    # "rating": age_rating
                })

            print(f"Found movies: {movie_info_list}")  # Debugging line
            return movie_title_list, movie_info_list
        
        except Exception as e:
            print(f"검색 중 오류가 발생했습니다: {str(e)}")
            return [], []

        finally:
            driver.quit() 

class CGVDetailCrawler:
    def __init__(self):
        self.driver = None
        
    def navigate_to_movie_detail_page(self, movie_title):
        self.driver = webdriver.Chrome()
        try:
            self.driver.get(CGV_SEARCH_URL + movie_title)
            
            wait = WebDriverWait(self.driver, 20)
            print("영화 상세 페이지로 이동중...")
            detail_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn_borderStyle1"))
            )
            detail_button.click()
            return True

        except Exception as e:
            print(f"영화 상세 페이지로 이동 중 오류가 발생했습니다: {e}")
            return False

    def fetch_plot_information(self):
        wait = WebDriverWait(self.driver, 20)
        print("줄거리 정보 크롤링중...")
        plot = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.sect-story-movie"))
        ).text
        return plot
    
    def fetch_reviews(self, review_limit=100):
            wait = WebDriverWait(self.driver, 20)
            print("리뷰 정보 크롤링중...")
            try:
                review_list = []
                page_no = 1

                while len(review_list) < review_limit:
                    try:
                        # Wait for reviews on the page
                        reviews = wait.until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".box-comment"))
                        )
                        # Extract review text
                        new_reviews = [review.text.strip() for review in reviews if review.text.strip()]
                        review_list += new_reviews[:review_limit - len(review_list)]  # Respect review limit

                        print(f"Page {page_no}: Fetched {len(new_reviews)} reviews.")

                        # Check for pagination
                        try:
                            # Locate the pagination container
                            pagination = self.driver.find_element(By.CSS_SELECTOR, "#paging_point")
                            page_links = pagination.find_elements(By.TAG_NAME, "a")                                                     # 페이지 번호 클릭
                            if page_no < len(page_links):  # Ensure the next page exists
                                next_page_link = pagination.find_element(By.LINK_TEXT, str(page_no + 1))
                                next_page_link.click()
                                time.sleep(2)  # Allow time for the next page to load
                                page_no += 1
                            elif page_no % 10 == 0:
                                # If the current page is a multiple of 10, click the "next 10 pages" button
                                next_button = pagination.find_element(By.XPATH, '//button[contains(@class, "btn-paging next")]')
                                if next_button.is_enabled():
                                    next_button.click()
                                    time.sleep(2)  # Allow time for the next page to load
                                    page_no += 1
                                else:
                                    print("Next button is disabled. Stopping.")
                                    break
                            else:
                                print("No more pages to navigate. Stopping.")
                                break
                        except NoSuchElementException:
                            print("No pagination or next button found. Ending review crawl.")
                            break

                    except Exception as e:
                        print(f"Error fetching reviews on page {page_no}: {e}")
                        break

                # Convert reviews to DataFrame and return
                movie_review_df = pd.DataFrame({"review": review_list[:review_limit]})
                return movie_review_df

            except Exception as e:
                print(f"리뷰 정보 크롤링 중 오류 발생: {e}")
                return pd.DataFrame({"review": []})
        
    def fetch_director_and_cast(self):
        wait = WebDriverWait(self.driver, 20)
        print("감독/출연진 정보 크롤링중...")
        try:
            cast_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='cast.aspx']"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", cast_button)
            time.sleep(2)  # Allow time for scrolling

            # Ensure no modals or overlays are blocking the element
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='cast.aspx']")))

            # Use JavaScript click to bypass obstructions
            self.driver.execute_script("arguments[0].click();", cast_button)
            time.sleep(2)  # Allow time for the cast details page to load

            # Extract directors
            director_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sect-staff-director ul li .box-contents dd a"))
            )
            directors = []
            for element in director_elements:
                # Split the text by newline and take the first part (the director's name)
                name = element.text.strip().split('\n')[0].strip()
                directors.append(name)

            # Extract actors
            actor_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sect-staff-actor ul li .box-contents dd a"))
            )
            actors = []
            for element in actor_elements:
                # Split the text by newline and take the first part (the actor's name)
                name = element.text.strip().split('\n')[0].strip()
                actors.append(name)

            return directors, actors
        except Exception as e:
            print(f"감독/출연진 정보 크롤링 중 오류 발생: {e}")
            return [], []
    


    def save_to_csv(self, movie_data, output_dir="data"):
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create or append to CSV file
            df = pd.DataFrame([movie_data])
            output_file = os.path.join(output_dir, "movie_details.csv")
            
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)
            
            print(f"데이터가 {output_file}에 저장되었습니다.")
            
        except Exception as e:
            print(f"CSV 저장 중 오류 발생: {e}")

    def crawl_movie_details(self, movie_title, review_limit=100):
        try:
            self.navigate_to_movie_detail_page(movie_title)
            plot = self.fetch_plot_information()
            reviews = self.fetch_reviews(review_limit)
            directors, cast_list = self.fetch_director_and_cast()

            # Join the directors and cast lists into strings
            directors_str = ', '.join(directors)  # Join directors with a comma
            cast_str = ', '.join(cast_list)        # Join cast with a comma

            movie_data = {
                "Title": movie_title,
                "Plot": plot,
                "Directors": directors_str,  # Use the joined string
                "Cast": cast_str,             # Use the joined string
                "Reviews": reviews
            }

            # self.save_to_csv(movie_data)
            return movie_data

        except Exception as e:
            print(f"Error occurred: {e}")
            return None

        finally:
            self.driver.quit() 