import nltk


def process_to_lstm(input_file, output_file):
    f = open(input_file)
    max_words = 0
    lines = f.readlines()
    proteins = set()
    for i in range(0, len(lines), 4):
        proteins.add(lines[i + 1].strip())
        proteins.add(lines[i + 2].strip())

    proteins = list(proteins)
    out = open(output_file, "w")
    for i in range(0, len(lines), 4):
        text = lines[i]
        protein1 = lines[i + 1]
        protein2 = lines[i + 2]
        label = lines[i + 3]

        text = text.replace("$T1$", "PROTEIN1")
        text = text.replace("$T2$", "PROTEIN2")

        words = nltk.word_tokenize(text)
        new_words = []
        for word in words:
            if word in proteins:
                new_words.append("PROTEIN")
            else:
                new_words.append(word)
            text = ' '.join(new_words)
        out.write(str(label).strip() + "\t" + text.strip() + "\n")

        words_num = len(words)
        if words_num > max_words:
            max_words = words_num

    print(max_words)

for i in range(1, 11):
    process_to_lstm("data/AiMed/folds/" + str(i) + "/train_pytorch_short_context.txt",
                    "data/AiMed/folds/" + str(i) + "/train_lstm.txt")
    process_to_lstm("data/AiMed/folds/" + str(i) + "/test_pytorch_short_context.txt",
                    "data/AiMed/folds/" + str(i) + "/test_lstm.txt")