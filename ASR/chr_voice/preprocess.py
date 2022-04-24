import os
import numpy as np

ROOT_DIR = "/ssd-playpen/home/shiyue/cherokee-audio-data"

colomns = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes',
           'age', 'gender', 'accent', 'locale', 'segment']
rows = []

dataset = "cno"
with open(f"{ROOT_DIR}/{dataset}/cno-training-data.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if "One's" in mp3:
            continue
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f'ffmpeg -i {ROOT_DIR}/{dataset}/{mp3} -c:a libmp3lame -q:a 2 -ac 1 clips/{path}')
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "durbin-feeling-tones"
with open(f"{ROOT_DIR}/{dataset}/data-chr.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "michael-conrad"
with open(f"{ROOT_DIR}/{dataset}/data-chr.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "michael-conrad2"
with open(f"{ROOT_DIR}/{dataset}/aligned.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "see-say-write"
with open(f"{ROOT_DIR}/{dataset}/data-chr.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "walc-1"
with open(f"{ROOT_DIR}/{dataset}/data-chr.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

dataset = "wwacc"
with open(f"{ROOT_DIR}/{dataset}/data-chr.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        client_id = f"{dataset}_{i}"
        items = line.strip().split('|')
        mp3 = items[-2]
        sentence = items[-1]
        if sentence == "x":
            continue
        path = f"{dataset}_{mp3.replace('/', '_')}"
        os.system(f"cp {ROOT_DIR}/{dataset}/{mp3} clips/{path}")
        rows.append([client_id, path, sentence, "2", "0", "", "", "", "chr"])

total = len(rows)
ten_percent = total // 10
print(total, ten_percent)

np.random.seed(42)
np.random.shuffle(rows)
test, dev, train = rows[:ten_percent], rows[ten_percent:ten_percent*2], rows[ten_percent*2:]

with open("train.tsv", 'w') as f:
    f.write('\n'.join(map(lambda x: '\t'.join(x), [colomns] + train)))
with open("test.tsv", 'w') as f:
    f.write('\n'.join(map(lambda x: '\t'.join(x), [colomns] + test)))
with open("dev.tsv", 'w') as f:
    f.write('\n'.join(map(lambda x: '\t'.join(x), [colomns] + dev)))
with open("other.tsv", 'w') as f:
    f.write('\n'.join(map(lambda x: '\t'.join(x), [colomns])))
with open("invalidated.tsv", 'w') as f:
    f.write('\n'.join(map(lambda x: '\t'.join(x), [colomns])))
