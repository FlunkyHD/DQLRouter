# Import of libraries needed for the service to run on the aSTEP website
from flask import Flask, jsonify, request, json
# Cors is used so the docker containers can to talk together
from flask_cors import CORS, cross_origin

# Basic configuration
service = Flask(__name__) #Service is the name we use to access the application as a whole.
cors = CORS(service)
service.config['CORS_HEADERS'] = 'Content-Type'

# The '/' is the index of the page.
@service.route('/')
def hello():
    return "Working as intended"

# The info tells where to place the service in the outer menu (category element) and what to name it.
@service.route('/info')
def info():
	return jsonify({
		'id': 'gruppe11-test-service',
		'name': 'Gruppe11GangGang',
		'version': '2021v1',
		'category': -1,
	})

# fields represents the input of the service. This is where input "parameters" are specified and named.
@service.route('/fields')
def fields():
	return jsonify({
		'user_fields': [
			{
				'type': 'input-number',
                'name': 'number1',
				'label': 'Insert first number',
				'placeholder': 'Input a number',
			},
            {
                'type': 'input-number',
                'name': 'number2',
                'label': 'Insert second number',
                'placeholder': 'Input a number',
            }
		],
		'developer_fields': [
		]
	})

# The readme is loaded when the documentation is loaded.
@service.route('/readme')
def readme():
    return jsonify({
		'chart_type': 'markdown',
		'content': ('Please insert two numbers, then press the blue button called "visualize results" to calculate the sum of the two numbers')
	})

# This function is called when the "visualize results" is pressed.
@service.route('/render', methods=['GET', 'POST'])
def render():
    number1 = request.form.get('number1')
    number2 = request.form.get('number2')
    
    if number2 is None or number1 is None or number1 == '' or number2 == '': 
        result = "One or more fields are empty"
    else:
        result = int(number1) + int(number2)
    
    return jsonify({
		'chart_type': 'markdown',
		'content': f"Resultat er: {result}"
	})

# This shows the raw data of the output in raw JSON format.
@service.route('/data', methods=['GET', 'POST'])
def data():
	return jsonify({
		'chart_type': 'markdown',
		'content': "Data example"
	})

# Combined endpoint, neeeded for notebook mode. Read the wiki for more information
@service.route('/combined', methods=['POST'])
def combine():
	return jsonify({
		'render': render(),
		'data': data()
	})

# Likewise for further information of the above used functions.
service.run(host="0.0.0.0", port=5000)
