from flask import Flask, render_template, abort
from flask.ext.pymongo import PyMongo
from calendar import Calendar
from datetime import date
import pymongo
import time
import datetime


app = Flask(__name__)

#mongo = PyMongo(app)
@app.route('/', defaults={'year': None})
@app.route('/<int:year>/')
#@app.route('/user/enirinth')

def index(year):
	conn = pymongo.MongoClient()
	db = conn.test
	cal = Calendar(0)
	try:
		if not year:
			year = date.today().year
		cal_list = [cal.monthdatescalendar(year, i+1) for i in xrange(12)]	
	except Exception, e:
		abort(404)
	else:
		records = [record for record in db.myCollections.find()]
		return render_template('cal.html', year=year, cal=cal_list, 
			                   events=records, cur_month=datetime.datetime.now().month)
	abort(404)

if __name__ == '__main__':
    app.run(debug=True)
    