# Makeup Recommender

This project is a recommender that will provide a one-stop shop experience where a user will get recommended an array of products to create an entire makeup look based on similar products that the user enjoys, products that similar users have purchased, as well as products that are personalized to the user including skin type, skin tone, allergies, and budget. Our project aims to utilize collaborative filtering recommendations along with content-based filtering to ensure user satisfaction and success when creating their desired look.


## How To Run

To install all dependencies, run the following command from the root directory of the project:
> ```pip install -r requirements.txt```

### Running the Project

To run the project, each command must start from the root directory of the project with:
> ```python run.py```

This base command can be modified with various different flags:
| Flag                | Type | Default Value             | Description                                                       |
|---------------------|------|---------------------------|-------------------------------------------------------------------|
| --item_data              |      |                      | Scrape item dataset from www.sephora.com.                              |
| --review_data              |      |                      | Scrape review dataset from www.sephora.com.                              |
| --features          |      |                           | Clean dataset and create features.                                |
| --model             |      |                           | Run model to create recommendations.                              |
| --accuracy          |      |                           | Evaluate accuracy of model.                                       |
| --test              |      |                           | Train model on a test dataset.                                    |
| --all               |      |                           | Train and evaluate accuracy of baseline model and our recommender model.      |


## Website

To visit the webpage copy and paste this URL into your browser: https://makeup-recommender.herokuapp.com/


## Document History

**Project Proposal**: https://docs.google.com/document/d/1bAXSUrQHcss8uU_eeqIJ4ewX3N-I8RLAjGJi77Zqw6c/edit

**Check-in**: https://docs.google.com/document/d/18rve8FbhRN8VXoO99ixB6nh96uWYtR-DNQMrjIwCr8Y/edit

**Report Checkpoint**: https://docs.google.com/document/d/1Xh3Ddskyy4niA7ZbzBUc9hoaR0bsrLKWgni9YzXTtZY/edit#

**Final Report**: https://docs.google.com/document/d/1WRj9ukUa-ozK3xtao-khCdradUQsR1k23tn4ELvQE3U/edit#

**Presentation Slides**: https://docs.google.com/presentation/d/1WYmy2IKTuVGE193Pq5t9v2BRTVCKxFHXy5zfLZN1K_I/edit?usp=sharing


## Credits

### For Usage of Scraping Script

https://github.com/jjone36/Cosmetic



### Responsibilities

**Alex Kim**:
* Scraped eye products and reviews
* Designed website
* Cleaned dataset
* Encode ingredient data
* Write rough draft of report (Abstract, Description of Data, Method, Metrics, Results)
* Deploy website on Heroku
* Use Flask to connect model to website (not implemented in final product)
* Fix UI with Streamlit
* Filter recommendations
* Edit report

**Justin Lee**:
* Fixed script to scrape data from Sephora
* Scraped lip products and reviews
* Coded data ingestion pipeline
* Incorporate run.py
* Code collaborative filtering model
* Create docker container
* Deploy website on Heroku
* Use Flask to connect model to website (not implemented in final product)
* Organize code and documentation

**Shayal Singh**:
* Fixed script to scrape data from Sephora
* Scraped face and cheek products and reviews
* Coded website layout
* Top popular baseline
* Write rough draft of report (Website, Conclusion)
* Deploy website on Heroku
* Use Flask to connect model to website (not implemented in final product)
* Display recommendations
* Visual presentation slides
* Fix UI with Streamlit
* Filter recommendations
* Finalize website
