import numpy as np
from Summarizer.Summarization import bart_summarizer, bert_summarizer, xlnet_summarizer, gpt2_summarizer, T5_summarizer
from KeywordExtraction.Keyword_Extractor import  Synonym_Keywords_Generation
from SpellCorrection.Spell_Correction import SpellCheck2
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    rawtext = request.form['rawtext']
    prediction_Spell = SpellCheck2(rawtext)
    prediction_Summ, Summ_Scores = bart_summarizer(prediction_Spell)
    scores , prediction_KeyWord,prediction_Synonyms = Synonym_Keywords_Generation(prediction_Spell)



    # output = round(prediction[0], 2)
    return render_template('index.html',ctext=rawtext, prediction_Spell='Did you mean: {}'.format(prediction_Spell),
    prediction_Summ = 'Summary: {}' .format(prediction_Summ),
    prediction_KeyWord= 'Keywords: {}' .format(prediction_KeyWord),
    prediction_Synonyms = 'Synonyms: {}' .format(prediction_Synonyms) )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction_Spell = SpellCheck2(data)
    prediction_Summ, Summ_Scores = bart_summarizer(prediction_Spell)
    scores , prediction_KeyWord, prediction_Synonyms = Synonym_Keywords_Generation(prediction_Spell)


    #rawtext = request.form['rawtext']
	#prediction = SpellCheck2(rawtext)
    # output = prediction
    return jsonify(prediction_Spell, prediction_Summ, prediction_KeyWord, prediction_Synonyms)

if __name__ == "__main__":
    app.run(debug=True)