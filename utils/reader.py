import csv

def read_tsv(file_name):
    rows = []
    index = 0
    with open(file_name, 'r', encoding='utf-8') as tsv:
        for line in csv.DictReader(tsv, delimiter='\t'):
            if(line['split'] == 'train'):
                first_sentence = line['sentence1']
                second_sentence = line['sentence2']
                label = line['label']
                rows.append([first_sentence, second_sentence, label])
                index += 1
    print(len(rows))
    return rows

if __name__ == "__main__":
    print(read_tsv("./data_raw/train.tsv")[:5])