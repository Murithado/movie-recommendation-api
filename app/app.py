from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app)

@app.route('/movie', methods=['GET'])
def recommend_movies():
    reslt = recommendation.results(request.args.get('title'))
    return jsonify(reslt)

