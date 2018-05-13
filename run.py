import lstm_api
results = lstm_api.get_sentiment("modi")
print (results)
print(type(results))
print(len(results))
print("%d %d"%(results[0],results[1]))
for i in results[2]:
	print("%s"%(i)) 

