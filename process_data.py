import json
from random import shuffle

import nltk
import os
import torch
from bioc import BioCXMLReader
from gensim.models import KeyedVectors


def is_in_relation(entity1, entity2, relations, entities_dict):
    for relation in relations:
        if entity1['id'] == relation['entity1'] and entity2['id'] == relation['entity2'] or \
            entity1['id'] == relation['entity2'] and entity2['id'] == relation['entity1']:
            return True
        elif entity1['text'] == entities_dict[relation['entity1']]['text'] and entity2['text'] == entities_dict[relation['entity2']]['text'] or \
                entity1['text'] == entities_dict[relation['entity2']]['text'] and entity2['text'] == entities_dict[relation['entity1']]['text']:
            return True
    return False

def is_in_one_sent(entity1, entity2, text):
    sentences = split_text_on_sentences(text)
    first_sent = -1
    second_sent = -1
    for s, i in zip(sentences, range(0, len(sentences))):
        if int(entity1['start']) >= s['start'] and int(entity1['start']) <= s['end']:
            first_sent = i
        if int(entity2['start']) >= s['start'] and int(entity2['start']) <= s['end']:
            second_sent = i
    if abs(first_sent-second_sent) <= 0 and first_sent != -1 and second_sent != -1:
        return True
    else:
        return False

def create_non_related_class(entities, relations, doc_id, text):
    num_rel = len(relations) + 1
    not_related = []
    entities_dict = load_entities_by_id(entities)
    for i in range(0, len(entities)):
        for j in range(i + 1, len(entities)):
            if not is_in_relation(entities[i], entities[j], relations, entities_dict) and is_in_one_sent(entities[i], entities[j], text) \
                and entities[i]['text'].lower() != entities[j]['text'].lower() and not is_in_relation(entities[i], entities[j], not_related, entities_dict):
                # print("")
                # print(entities[i]['text'])
                # print(entities[j]['text'])
                # print("")
                relation = {}
                relation['id'] = "R" + str(num_rel)
                relation['doc_id'] = doc_id
                relation['entity1'] = entities[i]['id']
                relation['entity2'] = entities[j]['id']
                relation['label'] = 0
                not_related.append(relation)
                num_rel += 1
    return not_related


def convert_data_to_json(corp_name):
    reader = BioCXMLReader(source="data/" + corp_name + "/" + corp_name.lower() + "_bioc.xml")
    out = open("data/" + corp_name + "/" + corp_name.lower() + "_json.txt", "w")
    reader.read()
    documents = reader.collection.documents
    rel_types = set()
    for document in documents:
        doc_id = document.id
        doc_text = document.passages[0].text
        doc_text = doc_text.replace('\n', ' ')

        annotations = document.passages[0].annotations
        entities = []
        for annotation in annotations:
            entity = {}
            entity['id'] = annotation.id
            entity['text'] = annotation.text
            entity['type'] = annotation.infons['type']
            entity['start'] = int(annotation.locations[0].offset)
            entity['end'] = entity['start'] + int(annotation.locations[0].length)
            entities.append(entity)

        relations = document.passages[0].relations
        doc_relations = []
        for relation in relations:
            doc_relation = {}
            if 'relation type' in relation.infons:
                doc_relation['type'] = relation.infons['relation type']
                rel_types.add(doc_relation['type'])
            else:
                doc_relation['type'] = ''
            doc_relation['doc_id'] = doc_id
            doc_relation['id'] = relation.id
            doc_relation['entity1'] = relation.nodes[0].refid
            doc_relation['entity2'] = relation.nodes[1].refid
            doc_relation['label'] = 1
            doc_relations.append(doc_relation)
        doc_relations.extend(create_non_related_class(entities, doc_relations, doc_id, doc_text))
        document_res = {}
        document_res['id'] = doc_id
        document_res['text'] = doc_text
        document_res['entities'] = entities
        document_res['relations'] = doc_relations
        out.write(json.dumps(document_res) + "\n")
    print(rel_types)

def convert_datasets_to_json():
    # corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    corpora = ['IEPA', 'HPRD50']
    for corp in corpora:
        convert_data_to_json(corp)

def load_entities_by_id(entities):
    entities_dict = {}
    for entity in entities:
        entities_dict[entity['id']] = entity
    return entities_dict

def cut_context(new_text):
    sentences = nltk.sent_tokenize(new_text)
    sent1 = -1
    sent2 = -1
    for (sent, i) in zip(sentences, range(0, len(sentences))):
        if 'entity1_term' in sent:
            sent1 = i
        if 'entity2_term' in sent:
            sent2 = i
    if sent1 == -1 or sent2 == -1:
        print("Sentences not found")
        print(new_text)
    if sent1 == sent2:
        return sentences[sent1]
    else:
        start_sent = min(sent1, sent2)
        end_sent = max(sent1, sent2)
        return ' '.join(sentences[start_sent: end_sent + 1])

def convert_data_to_ian_format(corp_name):
    f = open("data/" + corp_name + "/" + corp_name.lower() + "_json.txt")
    out = open("data/" + corp_name + "/" + corp_name.lower() + "_ian.txt", "w")
    for line in f:
        document = json.loads(line)
        text = document['text']
        entities = load_entities_by_id(document['entities'])
        relations = document['relations']
        for relation in relations:
            entity1 = entities[relation['entity1']]
            entity2 = entities[relation['entity2']]
            if entity1['start'] > entity2['start']:
                entity = entity1
                entity1 = entity2
                entity2 = entity
            start1 = int(entity1['start'])
            end1 = int(entity1['end'])
            start2 = int(entity2['start'])
            end2 = int(entity2['end'])
            new_text = text[0: start1] + 'entity1_term' + text[end1:start2] + 'entity2_term' + text[end2:]
            #print(new_text)
            # new_text = text[0: start1] + text[start - 1:].replace(entity1['text'], 'entity1_term')
            # start = int(entity2['start']) - len(entity1['text']) + len('entity1_term')
            # new_text = new_text[0: start - 1] + new_text[start - 1:].replace(entity2['text'], 'entity2_term')
            new_text = cut_context(new_text)
            out.write(new_text.strip() + "\n")
            out.write(entity1['text'] + "\n")
            out.write(entity2['text'] + "\n")
            out.write(str(relation['label']) + "\n")


def conver_datasets_to_ian_format():
    corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        convert_data_to_ian_format(corp)

def split_text_on_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sent_list = []
    pred_sent = 0
    for sent in sentences:
        sent_dict = {}
        sent_dict['text'] = sent
        sent_dict['start'] = text.index(sent, pred_sent)
        sent_dict['end'] = sent_dict['start'] + len(sent)
        sent_list.append(sent_dict)
        pred_sent = sent_dict['end']
    return sent_list

def count_statistic_of_intersentence_relation():
    # corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    corpora = ['BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        f = open("data/" + corp + "/" + corp.lower() + "_json.txt")
        count_in_sent = 0
        count_all = 0
        max_sent_dist = 0
        for line in f:
            document = json.loads(line)
            text = document['text']
            relations = document['relations']
            entities = load_entities_by_id(document['entities'])
            sentences = split_text_on_sentences(text)
            for relation in relations:
                if relation['label'] == 1:
                    entity1 = entities[relation['entity1']]
                    entity2 = entities[relation['entity2']]
                    if entity1['text'] == 'talin' and entity2['text'] == 'actin' and text == "In order to determine possible changes in the cytoskeleton organization and function during these processes, we have studied the in situ distribution of two major cytoskeleton-associated elements involved in the membrane anchorage of actin microfilaments, i.e. vinculin and talin, during the ontogeny of the neural crest and its derivatives in the avian embryo.":
                        print("Here")
                    first_sent = -1
                    second_sent = -1
                    for s, i in zip(sentences, range(0, len(sentences))):
                        if int(entity1['start']) >= s['start'] and int(entity1['start']) <= s['end']:
                            first_sent = i
                        if int(entity2['start']) >= s['start'] and int(entity2['start']) <= s['end']:
                            second_sent = i
                    if first_sent == second_sent and first_sent != -1 and second_sent != -1:
                        count_in_sent += 1
                    else:
                        if abs(first_sent - second_sent) > max_sent_dist:
                            max_sent_dist = abs(first_sent - second_sent)
                        if max_sent_dist == 1:
                            print(text)
                            print(entity1['text'])
                            print(entity2['text'])

                    count_all += 1
        print(corp)
        print("Max sent dist = " + str(max_sent_dist))
        print("In sentence = " + str(count_in_sent))
        print("All sentence = " + str(count_all))
        break

def fact(n):
    result = 0
    for i in range(1, n + 1):
        result *= i
    return result

def count_statistic():
    corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        f = open("data/" + corp + "/" + corp.lower() + "_json.txt")
        count_pos = 0
        count_neg = 0
        count_all = 0
        count_relations = 0
        for line in f:
            document = json.loads(line)
            relations = document['relations']
            entities = load_entities_by_id(document['entities'])
            entities_res = set()
            for relation in relations:
                if relation['label'] == 1:
                    count_pos += 1
                    entities_res.add(entities[relation['entity1']]['text'])
                    entities_res.add(entities[relation['entity2']]['text'])
                if relation['label'] == 0:
                    count_neg += 1
                count_all += 1
            # entities_text = set()
            # for entity in entities:
            #     entities_text.add(entity['text'])
            count_entities = len(entities_res)
            count_relations += (count_entities - 1) * count_entities / 2
        print(corp)
        print("Pos = " + str(count_pos))
        print("Neg = " + str(count_neg))
        print("All = " + str(count_all))
        print("Entities = " + str(count_relations))

def split_on_train_test_data():
    corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        f = open("data/" + corp + "/" + corp.lower() + "_ian.txt")
        out_train = open("data/" + corp + "/" + "train.txt", "w")
        out_test = open("data/" + corp + "/" + "test.txt", "w")
        lines = f.readlines()
        size = len(lines)
        train_size = int(size / 4 * 0.8) * 4
        train_dataset = lines[:train_size]
        test_dataset = lines[train_size:]
        out_train.writelines(train_dataset)
        out_test.writelines(test_dataset)

def check_datasets():
    corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        f = open("data/" + corp + "/train.txt")
        count_empty = 0
        for line in f:
            if line == "\n":
                count_empty += 1
        print(corp)
        print(count_empty)

def process_file_to_pytorch(lines, out):
    for i in range(0, len(lines), 4):
        text = nltk.word_tokenize(lines[i])
        text = ' '.join(text)
        text = text.replace('entity1_term', '$T1$')
        text = text.replace('entity2_term', '$T2$')
        out.write(text + "\n")
        out.write(lines[i + 1])
        out.write(lines[i + 2])
        out.write(lines[i + 3])

def process_data_to_ian_pytorch():
    corpora = ['AiMed', 'BioInfer', 'IEPA', 'HPRD50', 'LLL']
    for corp in corpora:
        f = open("data/" + corp + "/train.txt")
        out = open("data/" + corp + "/train_pytorch.txt", "w")
        lines = f.readlines()
        process_file_to_pytorch(lines, out)
        f.close()
        out.close()

        f = open("data/" + corp + "/test.txt")
        out = open("data/" + corp + "/test_pytorch.txt", "w")
        lines = f.readlines()
        process_file_to_pytorch(lines, out)
        f.close()
        out.close()

def count_context(filename):
    f = open(filename)
    lines = f.readlines()
    max_context = 0
    for i in range(0, len(lines), 4):
        text = lines[i]
        words = text.split(' ')
        length = len(words)
        if length > max_context:
            max_context = length
        if length > 300:
            print(text)
    print(max_context)

def split_on_balanced_folders(corpora):
    f = open("data/" + corpora + "train_pytorch.txt")
    lines = f.readlines()
    f.close()

    f = open("data/" + corpora + "test_pytorch.txt")
    lines.append(f.readlines())
    f.close()

def is_in_relation_cdr(entity1, entity2, relations):
    for r in relations:
        if r['entity1']['id'] == entity1['id'] and r['entity2']['id'] == entity2['id'] or \
                r['entity1']['id'] == entity2['id'] and r['entity2']['id'] == entity1['id']:
            return True
    return False

# def from_dict_to_str(entities):

def create_not_related(relations, entities, text):
    entities = list(entities.values())
    not_related = []
    for i in range(0, len(entities)):
        for j in range(i + 1, len(entities)):
            entity1 = entities[i]
            entity2 = entities[j]
            if entity1['start'] > entity2['start']:
                entity = entity1
                entity1 = entity2
                entity2 = entity
            if not is_in_relation_cdr(entity1, entity2, relations):
                r = {}
                r['entity1'] = entity1
                r['entity2'] = entity2
                r['label'] = 0
                r['text'] = text[0:entity1['start']] + "$T1$" + text[entity1['end']: entity2['start']] + "$T2$" + text[entity2['end']:]
                not_related.append(r)
    return not_related



def process_cdr_data(input_filename, output_file_name):
    f = open(input_filename)
    out = open(output_file_name, "w")
    reviews = []
    review_lines = []
    for line in f:
        if line == "\n":
            reviews.append(review_lines)
            review_lines = []
        else:
            review_lines.append(line)

    for review in reviews:
        text = review[0].split('|')[2].strip() + " "
        text += review[1].split('|')[2]
        entities = {}
        relations = []
        for i in range(2, len(review)):
            line_parts = review[i].strip().split("\t")
            if line_parts[1] != "CID":
                entity = {}
                entity['doc_id'] = line_parts[0]
                entity['start'] = int(line_parts[1])
                entity['end'] = int(line_parts[2])
                entity['text'] = line_parts[3]
                entity['type'] = line_parts[4]
                entity['id'] = line_parts[5].split("|")[0]
                entities[entity['id']] = entity

                if line_parts[5].find("|") != -1:
                    for i in range(0, line_parts[5].count('|')):
                        entity = {}
                        entity['doc_id'] = line_parts[0]
                        entity['start'] = int(line_parts[1])
                        entity['end'] = int(line_parts[2])
                        entity['text'] = line_parts[3]
                        entity['type'] = line_parts[4]
                        entity['id'] = line_parts[5].split("|")[i]
                        entities[entity['id']] = entity

            else:
                relation = {}
                relation['doc_id'] = line_parts[0]
                relation['type'] = line_parts[1]
                try:
                    entity1 = entities[line_parts[2]]
                    entity2 = entities[line_parts[3]]
                except:
                    continue
                if entity1['start'] > entity2['start']:
                    entity = entity1
                    entity1 = entity2
                    entity2 = entity
                relation['entity1'] = entity1
                relation['entity2'] = entity2
                relation['label'] = 1
                relation['text'] = text[0:entity1['start']] + "$T1$" + text[entity1['end']: entity2['start']] + "$T2$" + text[entity2['end']:]
                relations.append(relation)

        not_related = create_not_related(relations, entities, text)
        relations.extend(not_related)

        for r in relations:
            out.write(r['text'].strip() + "\n")
            out.write(r['entity1']['text'].strip() + "\n")
            out.write(r['entity2']['text'].strip() + "\n")
            out.write(str(r['label']).strip() + "\n")
    out.close()

def cut_the_context():
    f = open("data/CDR/development.txt")
    out = open("data/CDR/development_short1.txt", "w")
    lines = f.readlines()
    for i in range(0, len(lines), 4):
        text = lines[i]
        start = text.index("$T1$")
        end = text.index("$T2$") + 4
        text = text[start: end]
        sentences = nltk.sent_tokenize(text)
        if len(sentences) >= 2:
            text = sentences[0] + sentences[1]
        out.write(text + "\n")
        out.write(lines[i + 1])
        out.write(lines[i + 2])
        out.write(lines[i + 3])

def shuffle_list(data_list):
    indexes = [i for i in range(len(data_list))]
    shuffle(indexes)
    shuffeled_data_list = []
    for i in indexes:
        shuffeled_data_list.append(data_list[i])
    return shuffeled_data_list

def create_folds_cross_valid(corp_name):
    f = open("data/" + corp_name + "/" + corp_name.lower() + "_json.txt")
    os.mkdir("data/" + corp_name + "/folds")
    all_relations_1 = []
    all_relations_0 = []
    for line in f:
        document = json.loads(line)
        relations = document['relations']
        entities = load_entities_by_id(document['entities'])
        for r in relations:
            entity1 = entities[r['entity1']]
            entity2 = entities[r['entity2']]
            r['entity1'] = entity1
            r['entity2'] = entity2
            if r['label'] == 1:
                all_relations_1.append(r)
            elif r['label'] == 0:
                all_relations_0.append(r)
    f.close()

    rel_num_1 = len(all_relations_1)
    rel_num_0 = len(all_relations_0)
    rel_part_num_1 = int(rel_num_1 / 10)
    rel_part_num_0 = int(rel_num_0 / 10)
    for i in range(0, 10):
        os.mkdir("data/" + corp_name + "/folds/" + str(i + 1))
        rel_train_1 = all_relations_1[0: i * rel_part_num_1] + all_relations_1[(i + 1) * rel_part_num_1 + 1:]
        rel_test_1 = all_relations_1[i * rel_part_num_1: (i + 1) * rel_part_num_1]
        rel_train_0 = all_relations_0[0: i * rel_part_num_0] + all_relations_0[(i + 1) * rel_part_num_0 + 1:]
        rel_test_0 = all_relations_0[i * rel_part_num_0: (i + 1) * rel_part_num_0]

        rel_train = rel_train_1 + rel_train_0
        rel_test = rel_test_1 + rel_test_0
        rel_train = shuffle_list(rel_train)
        rel_test = shuffle_list(rel_test)

        f_train = open("data/" + corp_name + "/folds/" + str(i + 1) + "/train.txt", "w")
        for r in rel_train:
            f_train.write(json.dumps(r) + "\n")
        f_train.close()

        f_test = open("data/" + corp_name + "/folds/" + str(i + 1) + "/test.txt", "w")
        for r in rel_test:
            f_test.write(json.dumps(r) + "\n")
        f_test.close()

def count_class_statistic(corp_name):
    for i in range(0, 10):
        f_train = open("data/" + corp_name + "/folds/" + str(i + 1) + "/train.txt")
        f_test = open("data/" + corp_name + "/folds/" + str(i + 1) + "/test.txt")

        count_train_1 = 0
        count_train_0 = 0
        count_test_1 = 0
        count_test_0 = 0
        for line in f_train:
            relation = json.loads(line)
            if relation['label'] == 1:
                count_train_1 += 1
            elif relation['label'] == 0:
                count_train_0 += 1
        f_train.close()

        for line in f_test:
            relation = json.loads(line)
            if relation['label'] == 1:
                count_test_1 += 1
            elif relation['label'] == 0:
                count_test_0 += 1

        print(count_train_1)
        print(count_train_0)
        print(count_test_1)
        print(count_test_0)
        print("\n")

def process_fold_to_ian_pytorch(documents, input_filename, output_file_name):
    f = open(input_filename)
    out = open(output_file_name, "w")
    for line in f:
        relation = json.loads(line)
        entity1 = relation['entity1']
        entity2 = relation['entity2']
        text = documents[relation['doc_id']]['text']
        text = text[0:entity1['start']] + "$T1$" + text[entity1['end']: entity2['start']] + "$T2$" + text[entity2['end']:]
        out.write(text.strip() + "\n")
        out.write(entity1['text'].strip() + "\n")
        out.write(entity2['text'].strip() + "\n")
        out.write(str(relation['label']) + "\n")
    f.close()
    out.close()


def process_folds_to_ian_pytorch(corp_name):
    f = open("data/" + corp_name + "/" + corp_name.lower() + "_json.txt")
    documents = {}
    for line in f:
        document = json.loads(line)
        documents[document['id']] = document
    f.close()
    for i in range(0, 10):
        input_file_name = "data/" + corp_name + "/folds/" + str(i + 1) + "/train.txt"
        output_file_name = "data/" + corp_name + "/folds/" + str(i + 1) + "/train_pytorch.txt"
        process_fold_to_ian_pytorch(documents, input_file_name, output_file_name)

        input_file_name = "data/" + corp_name + "/folds/" + str(i + 1) + "/test.txt"
        output_file_name = "data/" + corp_name + "/folds/" + str(i + 1) + "/test_pytorch.txt"
        process_fold_to_ian_pytorch(documents, input_file_name, output_file_name)

def load_word2vec(path, word2idx = None):
    word_vec = {}
    w2v_model = KeyedVectors.load_word2vec_format(path)
    in_vocb = 0
    for word in w2v_model.vocab:
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = w2v_model[word]
            in_vocb += 1
    print("In vocabulary = %s, all words = %s", in_vocb, len(word2idx.keys()))
    return word_vec

fname = '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/'
word_vec = load_word2vec(fname)

# create_folds_cross_valid("LLL")
# count_class_statistic("LLL")

# process_folds_to_ian_pytorch("LLL")

# cut_the_context()



# process_cdr_data("/media/ilseyar/Disk_D/Ilseyar/Projects/CNN_CDR-master/data/Corpus/CDR_DevelopmentSet.PubTator.txt",
#                  "data/CDR/development.txt")






# count_context("data/CDR/train_short1.txt")
#
#
# process_data_to_ian_pytorch()
#
# convert_datasets_to_json()
# conver_datasets_to_ian_format()
# count_statistic_of_intersentence_relation()
# count_statistic()
# split_on_train_test_data()
#
# check_datasets()
#
# text = "this is the $T$ text"
# print(text.partition("$T$"))