"""from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))
corpus.print_summary_stats()"""
import json
from utils.load_jsonl import load_jsonl
conv_data = open('data/movie-corpus/conversations.json')
conv = json.load(conv_data)
utt = load_jsonl('data/movie-corpus/utterances.jsonl');
print(utt[0]['id'])
print (conv['L1044'])