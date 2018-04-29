from newsapi_fetcher import NewsFetcher
news_topic = input("Topic?: ")
News = NewsFetcher()
#News.set_start_date_from_now(2)
print (News.getStartDate())
print (News.getLastDate())
News.get_news(news_topic,sort='relevancy',domainName='thehindu.com')
f = open("info.text","w+")
for text in News.news:
	f.write(text+'\n')
f.close()

	
