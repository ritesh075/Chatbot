from flask import Flask, render_template, request
import model as mm

app=Flask(__name__, template_folder='template')
@app.route("/",method=["GET","POST"])
def home():
    if request.method=="POST":
        sentence=request.form["ask"]
        rply=mm.reply(sentence)
    return render_template('index.html', reply_txt=rply)

if __name__ == "__main__":
    app.run(debug=True)