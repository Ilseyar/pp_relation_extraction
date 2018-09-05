import json
import os


class RelationDataParser:

    def __init__(self, relation_types, max_possible_distance=2000):
        self.relation_types = relation_types
        self.max_possible_distance = max_possible_distance

    def is_file_contain_annotations(self, file):
        return file.find("ann") != -1

    def is_file_contain_text(self, file):
        return file.find("txt") != -1

    def get_file_id(self, file):
        return file[:-4]

    def is_relation(self, line):
        return line[0] == "R"

    def is_entity(self, line):
        return line[0] == "T"

    def get_entities_id(self, relation_line):
        relation_line_parts = relation_line.split(' ')
        return relation_line_parts[1].split(":")[1].strip(), \
               relation_line_parts[2].split(":")[1].strip()

    def is_in_relation_by_id(self, entity1, entity2, relation):
        return entity1['id'] == relation['entity1'] and entity2['id'] == relation['entity2']

    def is_in_relation_by_text(self, entity1, entity2, relation_entity_text1, relation_entity_text2):
        return entity1['text'] == relation_entity_text1['text'] and entity2['text'] == relation_entity_text2['text']

    def is_in_relations(self, entity1, entity2, relations, entities_dict):
        for relation in relations:
            if self.is_in_relation_by_id(entity1, entity2, relation) or self.is_in_relation_by_id(entity2, entity1, relation):
                return True
            else:
                relation_entity_text1 = entities_dict[relation['entity1']]
                relation_entity_text2 = entities_dict[relation['entity2']]
                if self.is_in_relation_by_text(entity1, entity2, relation_entity_text1, relation_entity_text2) or \
                    self.is_in_relation_by_text(entity2, entity1, relation_entity_text1, relation_entity_text2):
                    return True
        return False

    def is_texts_equal(self, entity1, entity2):
        return entity1['text'].lower() == entity2['text'].lower()

    def load_entities_by_id(self, entities):
        entities_dict = {}
        for entity in entities:
            entities_dict[entity['id']] = entity
        return entities_dict

    def get_entity_end_index(self, entity_line):
        end = entity_line.split(' ')[2]
        return int(end.split(';')[0])

    def is_in_acceptable_relation_type(self, entity1, entity2):
        type1 = entity1['type']
        type2 = entity2['type']
        if type1 + "-" + type2 in self.relation_types or \
                type2 + "-" + type1 in self.relation_types:
            return True
        else:
            return False

    def create_relation_type(self, entity1, entity2):
        if entity1['type'] == "Drug":
            return entity2['type'] + "-" + entity1['type']
        else:
            return entity1['type'] + "-" + entity2['type']

    def is_in_close_distance(self, entity1, entity2):
        if abs(entity1['start'] - entity2['start']) < self.max_possible_distance:
            return True
        else:
            return False

    # def is_short_context(self):


    def create_non_related(self, entities, relations, doc_id):
        num_rel = len(relations) + 1
        not_related = []
        entities_dict = self.load_entities_by_id(entities)
        for i in range(0, len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                if not self.is_texts_equal(entity1, entity2) \
                        and not self.is_in_relations(entity1, entity2, relations, entities_dict) \
                        and not self.is_in_relations(entity1, entity2, not_related, entities_dict) \
                        and self.is_in_acceptable_relation_type(entity1, entity2)\
                        and self.is_in_close_distance(entity1, entity2):
                    relation = {}
                    relation['id'] = "R" + str(num_rel)
                    relation['doc_id'] = doc_id
                    relation['entity1'] = entity1['id']
                    relation['entity2'] = entity2['id']
                    relation['type'] = self.create_relation_type(entity1, entity2)
                    relation['label'] = 0
                    not_related.append(relation)
                    num_rel += 1
        return not_related

    def convert_from_clinical_to_json(self, input_directory, output_directory):
        files = os.listdir(input_directory)
        out = open(output_directory, "w")
        data = {}
        i = 0

        for file in files:
            relations = []
            entities = []
            id = self.get_file_id(file)

            if id not in data:
                data[id] = {}
                data[id]['relations'] = []
            if self.is_file_contain_annotations(file):
                f = open(input_directory + "/" + file)
                for line in f:
                    line_parts = line.split("\t")

                    if self.is_relation(line):
                        relation = {}
                        entity1_id, entity2_id = self.get_entities_id(line_parts[1])
                        relation['entity1'] = entity1_id
                        relation['entity2'] = entity2_id
                        relation['type'] = line_parts[1].split(' ')[0]
                        relation['doc_id'] = id
                        relation['id'] = line_parts[0]
                        relation['label'] = 1
                        relations.append(relation)
                    elif self.is_entity(line):
                        entity = {}
                        entity['id'] = line_parts[0]
                        entity['start'] = int(line_parts[1].split(' ')[1])
                        entity['end'] = self.get_entity_end_index(line_parts[1])
                        entity['type'] = line_parts[1].split(' ')[0]
                        entity['text'] = line_parts[2].strip()
                        entities.append(entity)

                data[id]['relations'].extend(relations)
                data[id]['entities'] = entities
                f.close()

            # elif self.is_file_contain_text(file):
                f = open(input_directory + "/" + id + ".txt")
                text = f.readlines()
                text = ' '.join(text)
                not_related = self.create_non_related(data[id]['entities'],  data[id]['relations'], id)
                data[id]['relations'].extend(not_related)
                data[id]['text'] = text
                f.close()

            # if i == 1:
            #     break
                i += 1
                print(i)

        # out = open(output_directory, "w")
        # for id, value in data.items():
                value = data[id]
                document_res = {}
                document_res['id'] = id
                document_res['text'] = value['text']
                document_res['entities'] = value['entities']
                document_res['relations'] = value['relations']
                out.write(json.dumps(document_res) + "\n")

        out.close()

