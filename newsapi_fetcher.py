from newsapi_client import NewsApiClient;
from datetime import datetime, timedelta;


class NewsFetcher():
    'Fetches News for specified topics and parameters'

    def __init__(self, key='016fad42f4c149169644920d70a1a036'):
        self.newsApi = NewsApiClient(api_key=key)
        self.LastDate = datetime.today().date()
        self.StartDate = None
        self.news = []

    def getStartDate(self):
        return self.StartDate

    def getLastDate(self):
        return self.LastDate

    def set_start_date_from_now(self, no_of_weeks):
        self.LastDate = datetime.today().date()
        if no_of_weeks > 4:
            no_of_weeks = 4
        self.StartDate = self.LastDate - timedelta(days=7 * no_of_weeks);

    def get_news(self, topic, domainName=None, sort='relevancy'):
        first_page = self.newsApi.get_everything(q=topic,from_parameter=self.StartDate,domains=domainName,
                                                 to=self.LastDate, language='en', sort_by=sort, 							page=1)

        print ("Total Hits: %d"%first_page['totalResults'])
        i=1;
        for article in first_page['articles']:
            #print (str(i)+' '+article['source']['name']+'\n'+article['title'])
            #print ('\n'+article['url']+' '+article['publishedAt']+'\n')
            i=i+1
            #self.news.append(article['title']+article['descriptions'])
            #print(article['title']+article['description'])
            self.news.append(article['title'])
			



