# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:30:01 2020

@author: rahul
"""

from transformers import pipeline
from summarizer import Summarizer, TransformerSummarizer
from rouge import Rouge
import pprint

rouge = Rouge()

def bart_summarizer(data):
    summarizer_bart = pipeline(task='summarization', model="bart-large-cnn")
    summary_bart = summarizer_bart(data, min_length=30, max_length = 140)
    print('Bart for Text - Summarization')
    summary = summary_bart[0]['summary_text']
    rouge_scores = rouge.get_scores(summary, data)
    return summary, rouge_scores

def bert_summarizer(data):
    summarizer_bert = Summarizer()
    summary_bert = summarizer_bert(data, min_length=30, max_length = 140)
    summary = ''.join(summary_bert)
    rouge_scores = rouge.get_scores(summary, data)
    return summary, rouge_scores
    

def xlnet_summarizer(data):
    summarizer_xlnet = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    print('\n XLNet for Text - Summarization')
    summary_xlnet = summarizer_xlnet(data, min_length=30, max_length = 140)
    summary = ''.join(summary_xlnet)
    rouge_scores = rouge.get_scores(summary, data)
    return summary, rouge_scores

def gpt2_summarizer(data):
    summarizer_gpt2 = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    summary_gpt2 = summarizer_gpt2(data, min_length=20, max_length = 140)
    summary = ''.join(summary_gpt2)
    rouge_scores = rouge.get_scores(summary, data)
    return summary, rouge_scores

def T5_summarizer(data):
    summarizer_t5 = pipeline(task='summarization', model="t5-large")
    print('\n T5 for Text - Summarization')
    summary_t5 = summarizer_t5(data, min_length=30, max_length = 140) # change min_ and max_length for different output
    summary = summary_t5[0]['summary_text']
    rouge_scores = rouge.get_scores(summary, data)
    return summary, rouge_scores
    


data = '''
Traditional electronics are connected with wires to the power supply to switches and sensors. Over the time the amount of sensors and IoT devices is growing so there is no more space in future for all the cables needed for the connections.
MID (molded interconnect devices) opens the possibility to place tracks directly on the surface of plastic parts.
MID must be further developed for our own applications. E.g. printing in plastic housing parts.
We expect that this will be the enabler for much more sensors in our tools. 5 years ago the MID technology was not competitive vs. classical cabling/wiring. With increasing electronics, cabling space gets rare. What technologies and companies exist on the market to create MID? Interesting technologies could be laser activation, galvanization, deformable sheets.
'''

# summary,scores = bart_summarizer(data)
# print(summary)
# print(scores)
