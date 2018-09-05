import json
import os

from RelationDataParser import RelationDataParser
# from process_data import create_non_related_class

DATA_PATH = "data/n2c2/track2-training_data_2"
ACCEPTABLE_RELATIONS = ["Strength-Drug",
                        "Dosage-Drug",
                        "Duration-Drug",
                        "Frequency-Drug",
                        "Form-Drug",
                        "Route-Drug",
                        "Reason-Drug",
                        "ADE-Drug"]
MAX_DISTANCE = 2023
DATA_JSON_PATH = "data/n2c2/n2c2.json"


def find_unused_parts():
    files = os.listdir(DATA_PATH)
    text_ann_dict = {}
    for file in files:
        if file[:-4] not in text_ann_dict:
            text_ann_dict[file[:-4]] = {}
        if file.find("txt") != -1:
            f = open(DATA_PATH + "/" + file)
            text = ' '.join(f.readlines())
            text_ann_dict[file[:-4]]['text'] = text
        elif file.find("ann") != -1:
            f = open(DATA_PATH + "/" + file)
            entities = []
            for line in f:
                line_parts = line.split("\t")
                if line_parts[0].find('T') != -1:
                    entity ={}
                    entity['start'] = int(line_parts[1].split(' ')[1])
                    entity['end'] = int(line_parts[1].split(' ')[2].split(';')[0])
                    entity['text'] = line_parts[2]
                    entities.append(entity)
            text_ann_dict[file[:-4]]['entities'] = entities

    count_start = 0
    count_end = 0
    count_no_start = 0
    count_no_end = 0
    for k in text_ann_dict.keys():
        v = text_ann_dict[k]
        text = v['text']
        entities = v['entities']
        start = 0
        end = len(text)
        try:
            start = text.index("Service:")
        except:
            count_no_start += 1
        try:
            end = text.index("Department:")
        except:
            count_no_end += 1
        for entity in entities:
            if entity['start'] < start:
                print(k)
                # print(entity['text'])
                # count_start += 1
            if entity['end'] > end:
                count_end += 1
                print(k)
                print(entity['text'])

    print(count_start)
    print(count_end)
    print(count_no_start)
    print(count_no_end)

def count_startistic():
    files = os.listdir(DATA_PATH)
    max = 0
    for file in files:
        relations = []
        entities = {}
        if file.find("ann") != -1:
            f = open(DATA_PATH + "/" + file)
            for line in f:
                line_parts = line.split("\t")
                if line_parts[0].find("R") != -1:
                    relation = {}
                    entity1 = line_parts[1].split(' ')[1].split(":")[1]
                    entity2 = line_parts[1].split(' ')[2].split(":")[1].strip()
                    relation['entity1'] = entity1
                    relation['entity2'] = entity2
                    relations.append(relation)
                elif line_parts[0].find("T") != -1:
                    entities[line_parts[0]] = int(line_parts[1].split(' ')[1])
        for relation in relations:
            entity1 = entities[relation['entity1']]
            entity2 = entities[relation['entity2']]
            if abs(entity1-entity2) > max:
                max = abs(entity1-entity2)
            # if abs(entity1-entity2) == 2023:
            #     print(file)
            #     print(relation)
    print(max)

# def process_data_to_json():
#     files = os.listdir(DATA_PATH)
#     data = {}
#     for file in files:
#         relations = []
#         entities = []
#         if file[:-4] not in data:
#             data[file[:-4]] = {}
#         if file.find("ann") != -1:
#             f = open(DATA_PATH + "/" + file)
#             for line in f:
#                 line_parts = line.split("\t")
#                 if line_parts[0].find("R") != -1:
#                     relation = {}
#                     entity1 = line_parts[1].split(' ')[1].split(":")[1]
#                     entity2 = line_parts[1].split(' ')[2].split(":")[1].strip()
#                     relation['entity1'] = entity1
#                     relation['entity2'] = entity2
#                     relation['type'] = line_parts[1].split(' ')[0]
#                     relation['doc_id'] = file[:-4]
#                     relation['id'] = line_parts[0]
#                     relation['label'] = 1
#                     relations.append(relation)
#                 elif line_parts[0].find("T") != -1:
#                     entity = {}
#                     entity['id'] = line_parts[0]
#                     entity['start'] = int(line_parts[1].split(' ')[1])
#                     entity['end'] = int(line_parts[1].split(' ')[2])
#                     entity['type'] = line_parts[1].split(' ')[0]
#                     entity['text'] = line_parts[2].strip()
#                     entities.append(entity)
#             data[file[:-4]]['relations'] = relations
#             data[file[:-4]]['entities'] = entities
#         elif file.find("txt") != -1:
#             f = open(DATA_PATH + "/" + file)
#             text = f.readlines()
#             text = ' '.join(text)
#             id = file[:-4]
#             not_related = create_non_related_class(entities, relations, id, text)
#             relations.extend(not_related)

def get_entities_by_type(type, entities):
    result_entities = []
    for entity in entities:
        # if entity['type'] in type:
            result_entities.append(entity)
    return result_entities

def load_entities_by_id(entities):
    entities_dict = {}
    for entity in entities:
        entities_dict[entity['id']] = entity
    return entities_dict

def find_entity_in_list(entity, entity_list):
    index = 0
    for e in entity_list:
        if e['id'] == entity['id'] and e['start'] == entity['start']:
            return index
        index += 1

def check_nearest_dosage():
    f = open("data/n2c2/n2c2.json")
    count = 0
    max_dos = 0
    max_drug = 0
    for line in f:
        document = json.loads(line)
        relations = document['relations']
        entities = document['entities']
        entities = get_entities_by_type(["ADE", "Drug"], entities)
        entities = sorted(entities, key=lambda k: k['start'])
        entities_dict = load_entities_by_id(entities)
        for relation in relations:
            if relation['label'] == 1 and relation['entity1'] in entities_dict and relation['entity2'] in entities_dict:
                entity1 = entities_dict[relation['entity1']]
                entity2 = entities_dict[relation['entity2']]
                if entity2['type'] == "Drug":
                    entity = entity1
                    entity1 = entity2
                    entity2 = entity
                index1 = find_entity_in_list(entity1, entities)
                index2 = find_entity_in_list(entity2, entities)
                if abs(index2-index1) != 1:
                    count_drug = 0
                    for ent in entities[min(index1, index2): max(index1, index2)]:
                        if ent['type'] == "Drug":
                            count_drug += 1

                    if count_drug > max_drug:
                        max_drug = count_drug
                    count += 1
                    # print(abs(index2-index1))
                    if abs(entity1['start'] - entity2['start']) > max_dos:
                        max_dos = abs(entity1['start'] - entity2['start'])
                    # print(relation['doc_id'])
                    # print(entity1['text'])
                    # print(entity2['text'])
                    # print(relation['id'])
    print(count)
    print(max_dos)
    print(max_drug)

def process_text_parts(text_parts, text):
    start = 0
    result = []
    for text_part in text_parts:
        # print(text_part)
        # print(text)
        start = text.index(text_part.strip(), start)
        end = start + len(text_part)
        result.append({"start": start, "end": end, "text": text_part})
    return result

def count_class_statistic():
    f = open(DATA_JSON_PATH)
    count_label_0 = 0
    count_label_1 = 0
    count_dist = 0
    count_dif_parts = 0
    for line in f:
        document = json.loads(line)
        relations = document['relations']
        entities = load_entities_by_id(document['entities'])
        text = document['text']
        text_parts = text.split("\n \n")
        text_parts = process_text_parts(text_parts, text)
        print(len(text_parts))
        for r in relations:
            if r['label'] == 1:
                count_label_1 += 1
            else:
                count_label_0 += 1
            entity1 = entities[r['entity1']]
            entity2 = entities[r['entity2']]
            if r['label'] == 1:
                is_in_one_part = False
                for text_part in text_parts:
                    if entity1['start'] > text_part['start'] and entity1['end'] < text_part['end'] \
                        and entity2['start'] > text_part['start'] and entity2['end'] < text_part['end']:
                        is_in_one_part = True
                if not is_in_one_part:
                    count_dif_parts += 1
            # if abs(entity1['start']-entity2['start']) > 200:
            #     count_dist += 1


    print(count_label_1)
    print(count_label_0)
    print(count_dist)
    print(count_dif_parts)

# count_class_statistic()

# find_unused_parts()
# count_startistic()
# relationDataParser = RelationDataParser(ACCEPTABLE_RELATIONS, MAX_DISTANCE)
# relationDataParser.convert_from_clinical_to_json(DATA_PATH, "data/n2c2/n2c2.json")

check_nearest_dosage()

# f = open("data/n2c2/track2-training_data_2/100035.txt")
# text = ''.join(f.readlines())
# print(text.index("topiramate"))
# print(text[16357:16492])

