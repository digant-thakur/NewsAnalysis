from newsapi_client import NewsApiClient;
from datetime import datetime, timedelta;


class NewsFetcher:
    'Fetches News for specified topics and parameters'

    def __init__(self, key='016fad42f4c149169644920d70a1a036'):
        self.newsApi = NewsApiClient(api_key=key)
        self.LastDate = datetime.today().date()
        self.StartDate = None
        self.news = {}
        self.urlToImage = None

    def getStartDate(self):
        return self.StartDate

    def getLastDate(self):
        return self.LastDate

    def set_start_date_from_now(self, no_of_weeks):
        self.LastDate = datetime.today().date()
        if no_of_weeks > 4:
            no_of_weeks = 4
        self.StartDate = self.LastDate - timedelta(days=7 * no_of_weeks);

    def get_news_dict(self):
        return self.news

    def get_cover_image(self):
        return self.urlToImage

    def get_news(self, topic, domainName=None, sort='relevancy'):
        first_page = self.newsApi.get_everything(q=topic, from_parameter=self.StartDate, domains=domainName,
                                                 to=self.LastDate, language='en', sort_by=sort, page=1)
        self.urlToImage = first_page['articles'][0]['urlToImage']
        hits = first_page['totalResults']
        print ("Total Hits: %d" % hits)
        for article in first_page['articles']:
            self.news[article['title'] + article['description']] = article['url']
        for num in range(2, int(hits / 100)):
            page = self.newsApi.get_everything(q=topic, from_parameter=self.StartDate, domains=domainName,
                                               to=self.LastDate, language='en', sort_by=sort, page=num)
            for article in page['articles']:
                self.news[article['title'] + article['description']] = article['url']
