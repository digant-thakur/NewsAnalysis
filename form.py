from flask import Flask,redirect,url_for,request,render_template
import lstm_api
app = Flask(__name__)



@app.route('/success/<value>')
def success(value):
	results = lstm_api.get_sentiment(value)
	return render_template('output.html',results = results)
	
@app.route('/search',methods = ['POST','GET'])
def search():
	if request.method =='POST':
		user = request.form['nm']
		return redirect(url_for('success',value = user))
	else:
		user = request.args.get('nm')
		return redirect(url_for('success',value = user))
		
if __name__=="__main__":
	app.run(debug = True)


