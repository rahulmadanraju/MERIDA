# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:13:04 2020

@author: rahul
"""

from Summarizer.Summarization import bart_summarizer, bert_summarizer, xlnet_summarizer, gpt2_summarizer, T5_summarizer
from KeywordExtraction.Keyword_Extractor import *


data = '''
Traditional electronics are connected with wires to the power supply to switches and sensors. Over the time the amount of sensors and IoT devices is growing so there is no more space in future for all the cables needed for the connections.
MID (molded interconnect devices) opens the possibility to place tracks directly on the surface of plastic parts.
MID must be further developed for our own applications. E.g. printing in plastic housing parts.
We expect that this will be the enabler for much more sensors in our tools. 5 years ago the MID technology was not competitive vs. classical cabling/wiring. With increasing electronics, cabling space gets rare. What technologies and companies exist on the market to create MID? Interesting technologies could be laser activation, galvanization, deformable sheets.
'''

# summarization
summary,scores = bart_summarizer(data)
print(summary)
print(scores)

summary,scores = bert_summarizer(data)
print(summary)
print(scores)

summary,scores = xlnet_summarizer(data)
print(summary)
print(scores)

summary,scores = gpt2_summarizer(data)
print(summary)
print(scores)

#summary,scores = T5_summarizer(data)
#print(summary)
#print(scores)

# Keyword extraction
Keywords, feature_names = tfidf_Data(data, 0, 1000, 1,3)
sorted_items = sort_coo(Keywords.tocoo())
Keywords = extract_topn_from_vector(feature_names,sorted_items,10)
print(Keywords)