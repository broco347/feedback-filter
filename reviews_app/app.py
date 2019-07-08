from flask import Flask, render_template, jsonify, request, abort
from api import find_clusters

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cluster", methods=['POST'])
def cluster():
    print("In cluster")
    if not request.json:
        abort(400)
    data = request.json
    print(data)
    feedback = data['feedback']
    # return jsonify({
    #     "title1": ["aoij", "oiajsd", "aosidja"],
    #     "another title": ["Hey its some more text"]
    # })

    response = find_clusters(feedback)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
