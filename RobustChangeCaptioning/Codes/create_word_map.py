import json
import os

LOCATION = "/content/"
INCLUDE_SPLITS = ['train', 'val']


all_words = []

for split in INCLUDE_SPLITS:
    data = json.load(open(os.path.join(LOCATION, split+".json"), "r"))
    for instance in data:
        for s in instance['sentences']:
            all_words.extend(s.split(" "))

word_map = {}

idx = 3
for w in all_words:
    if w not in word_map:
        word_map[w] = idx
        idx += 1

print(len(word_map))

json.dump(word_map, open("/content/word_map.json", "w"))