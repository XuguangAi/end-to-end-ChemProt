# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:06:22 2022

@author: Xuguang Ai
"""

import pandas as pd
import re
import json
import stanza
from tqdm import tqdm
from collections import Counter

Path = '.../ChemProt_Corpus/chemprot_training' # Path to the training dataset folder

nlp = stanza.Pipeline(lang='en', processors='tokenize')

lookup = {"GENE-Y": "GENE", "GENE-N": "GENE", "CHEMICAL": "CHEMICAL"}

df_abstracts = pd.read_table(Path + "/chemprot_training_abstracts.tsv", header=None, keep_default_na=False, names=["doc_key", "title", "abstract"], encoding='utf-8')

df_entities = pd.read_table(Path + "/chemprot_training_entities.tsv", header=None, keep_default_na=False, names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')

df_relations = pd.read_table(Path + "/chemprot_training_gold_standard.tsv", header=None, keep_default_na=False, names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

# ******** Modify start and end position indices of incorrect annotations ********

for i in range(1594, 1642):
    if i == 1616:
        df_entities.at[i,'char_start'] += 1
        df_entities.at[i,'char_end'] += 1
    if i == 1636:
        pass
    else:
        df_entities.at[i,'char_start'] -= 2
        df_entities.at[i,'char_end'] -= 2
# Almost all entities in this abstract (doc_key = 23194825) need a shift of 1 or 2 spaces.

# ******** Only specific to BioCreative VI ChemProt training dataset ********

def format_relations(relations):
    # Convert to dictionaries
    res = {}
    for _, row in relations.iterrows():
        ent1 = row["arg1"].replace("Arg1:", "")
        ent2 = row["arg2"].replace("Arg2:", "")
        key = (ent1, ent2)
        res[key] = row["label"]
    return res


def get_relations_in_sent(aligned, relations):
    res = []
    # Loop over the relations and keep the ones relating entities in this sentence
    for ents, label in relations.items():
        if ents[0] in aligned and ents[1] in aligned:
            ent1 = aligned[ents[0]]
            ent2 = aligned[ents[1]]
            to_append = ent1[:2] + ent2[:2] + (label,)
            res.append(to_append)
    return res


def create_coarse_toks(sent):
    return sent.split()
    # Tokenized by spaces


def create_word_len_dict(sent):
    # Create a dictionary with key: coarse token indices and value: coarse token length
    coarse_toks = create_coarse_toks(sent)

    word_len_dict = {}
    for i in range(len(coarse_toks)):
        word_len_dict[i] = len(coarse_toks[i])
    return word_len_dict


def create_start_char_dict(sent):
    # Create a dictionary with key: coarse token indices and value: coarse token start positions
    coarse_toks = create_coarse_toks(sent)
    word_len_dict = create_word_len_dict(sent)
    start_char_dict = {}

    for i in range(len(coarse_toks)):
        if i == len(coarse_toks) - 1:
            iter = re.finditer(r'' + re.escape(coarse_toks[i]) + r'', sent)
            indices = [m.start(0) for m in iter]
            if len(indices) == 1:
                start_char_dict[i] = indices[0]
            else:
                start_char_dict[i] = indices
        else:
            iter = re.finditer(r'' + re.escape(coarse_toks[i]) + r' ', sent)
            indices = [m.start(0) for m in iter]
            if len(indices) == 1:
                start_char_dict[i] = indices[0]
            else:
                start_char_dict[i] = indices

    # Example: start_char_dict[0] = 0. start_char_dict[1] = [7, 168]. That is because
    # the 1st token was matched with the token with start positions 7 or 168.
    # We will choose the correct start position as below (which is 7 in this case):

    for i in range(len(start_char_dict)):
        if type(start_char_dict[i]) is list:
            if i == 0:
                start_char_dict[i] = start_char_dict[i][0]
            elif i == len(start_char_dict) - 1:
                start_char_dict[i] = start_char_dict[i][-1]
            else:
                for j in range(len(start_char_dict[i])):
                    temp = start_char_dict[i][j]
                    if temp > start_char_dict[i - 1] + word_len_dict[i - 1]:
                        # We choose the start position as small as possible with the condition that
                        # the start position is larger than the start position of previous token plus
                        # the length of previous token.
                        # Example: 7 > 0 + word_len_dict[0] = 0 + 6 so we choose 7 from [7, 168]
                        start_char_dict[i] = temp
                        break

    return start_char_dict


def create_end_char_dict(sent, start_char_dict):
    word_len_dict = create_word_len_dict(sent)

    end_char_dict = {}
    for i in range(len(start_char_dict)):
        end_char_dict[i] = start_char_dict[i] + word_len_dict[i]
    return end_char_dict
    # Given start positions of coarse tokens, we can calculate end positions of coarse tokens by simply
    # adding lengths of coarse tokens


def start_char(n, len_dict):        
    if n == 0:
        return 0     
    if n == 1:
        return len_dict[0]  
    else:       
        return len_dict[n-1] + start_char(n-1, len_dict)


def end_char(n, len_dict):           
    if n == 0:
        return len_dict[0]
    else:       
        return len_dict[n] + end_char(n-1, len_dict)


# We create a dictionary with key: grained tokens and value: start and end positions of grained tokens
# By grained tokens we mean coarse token 'Na+CL-' was further tokenized as 'Na', '+', 'Cl' and '-'.
# This can deal with tricky chemical or disease entities which cannot be properly tokenized by known tokenizers 
def create_dict_indices(sent, start_char_dict, end_char_dict):
    coarse_toks = sent.split()

    dict_indices = {}
    for i in range(len(coarse_toks)):
        dict_indices[i] = [coarse_toks[i], [start_char_dict[i], end_char_dict[i]]]

    for i in range(len(coarse_toks)):
        if coarse_toks[i] != re.findall(r"[A-Za-zα-ωΑ-Ω]+|\d+|[^A-Za-zα-ωΑ-Ω\s]", coarse_toks[i])[0]:
            # If the token can be further tokenized
            dict_indices[i][0] = re.findall(r"[A-Za-zα-ωΑ-Ω]+|\d+|[^A-Za-zα-ωΑ-Ω\s]", coarse_toks[i])

    for i in range(len(dict_indices)):
        if type(dict_indices[i][0]) == list:
            # The list contains grained tokens
            start_pos = dict_indices[i][1][0]
            dict_indices[i][1] = []
            len_dict = {}
            for l in range(len(dict_indices[i][0])):
                len_dict[l] = len(dict_indices[i][0][l])
            for k in range(len(dict_indices[i][0])):
                list_temp = []
                list_temp.append(start_char(k, len_dict) + start_pos)
                list_temp.append(end_char(k, len_dict) + start_pos)
                dict_indices[i][1].append(list_temp)

    return dict_indices
    # Example: dict_indices[14] = [['partner', "'", 's'], [[76, 83], [83, 84], [84, 85]]]


def create_dict_final(sent, dict_indices):
    # We create a dictionary with key: grained tokens and value: start and end positions of grained tokens
    # without grouping coarse tokens
    toks_list = []
    for i in range(len(dict_indices)):
        if type(dict_indices[i][0]) == str:
            toks_list.append(dict_indices[i][0])
        if type(dict_indices[i][0]) == list:
            for j in range(len(dict_indices[i][0])):
                toks_list.append(dict_indices[i][0][j])

    toks_indices_list = []
    for i in range(len(dict_indices)):
        if type(dict_indices[i][0]) == str:
            toks_indices_list.append(dict_indices[i][1])
        if type(dict_indices[i][0]) == list:
            for j in range(len(dict_indices[i][1])):
                toks_indices_list.append(dict_indices[i][1][j])

    dict_final = {}
    for i in range(len(toks_indices_list)):
        dict_final[i] = [toks_list[i], toks_indices_list[i]]

    return dict_final
    # Example: dict_final[11] = ['partner', [76, 83]]. dict_final[12] = ["'", [83, 84]]
    # dict_final[13] = ['partner', [84, 85]]


def one_abstract(abstract, df_entities_FH, df_relations_FH):
    # Convert one abstract into scierc format

    num_aligned = 0

    doc = abstract["title"] + " " + abstract["abstract"]
    unicode = ['\xa0', '\u2002', '\u2005', '\u2009', '\u200a']
    for item in unicode:
        doc = doc.replace(item, u' ')
    processed = nlp(doc)
    doc_key = abstract["doc_key"]
    entities = df_entities_FH[df_entities_FH['doc_key'] == doc_key]
    relations = format_relations(df_relations_FH[df_relations_FH['doc_key'] == doc_key])

    scierc_format = {"doc_key": doc_key, "sentences": [], "ner": [], "relations": []}

    sent_len_dict = {}
    for i in range(len(processed.sentences)):
        sent_len_dict[i] = len(processed.sentences[i].text)

    start_sent = {}
    for i in range(len(processed.sentences)):
        start_sent[i] = doc.index(processed.sentences[i].text)

    end_sent = {}
    for i in range(len(processed.sentences)):
        end_sent[i] = doc.index(processed.sentences[i].text) + sent_len_dict[i]

    span_num_list = [0]


    for i in range(len(processed.sentences)):
        sent = processed.sentences[i].text

        # Align entities in a sentence
        start, end = start_sent[i], end_sent[i]
        start_ok = entities["char_start"] >= start
        end_ok = entities["char_end"] <= end
        keep = start_ok & end_ok
        entities_sent = entities[keep]

        # Obtain start and end positions of grained tokens
        start_char_dict = create_start_char_dict(sent)
        end_char_dict = create_end_char_dict(sent, start_char_dict)
        dict_indices = create_dict_indices(sent, start_char_dict, end_char_dict)
        dict_final = create_dict_final(doc, dict_indices)

        if span_num_list == [0]:
            span_num_list.append(len(dict_final))
        else:
            span_num_list.append(len(dict_final) + span_num_list[-1])

        if entities_sent.empty:
            aligned = {}

        else:
            aligned_entities = {}

            for _, row in entities_sent.iterrows():

                start_tok, end_tok = None, None
                start_num, end_num = 0, 0
                start_span, end_span = 0, 0

                for i in range(len(dict_final)):
                    if dict_final[i][1][0] + start == row["char_start"]:
                        start_tok = dict_final[i][0]
                        start_num += dict_final[i][1][0]
                        start_span += i
                    if dict_final[i][1][1] + start == row["char_end"]:
                        end_tok = dict_final[i][0]
                        end_num += dict_final[i][1][1]
                        end_span += i

                expected = sent[start_num:end_num]

                if start_tok is None or end_tok is None:
                    print('Real entity start and end:', row["char_start"], row["char_end"], row.text)
                    print('False entity start and end:', start_tok, end_tok)
                    print('Context:', sent[row["char_start"] - 20:row["char_end"] + 20])
                    print('doc_key', doc_key)
                    COUNTS['count_missed_type_1'] += 1


                elif expected != row.text:
                    print('Real entity:', row.text)
                    print('False entity:', expected)
                    print('Context:', doc[row["char_start"] - 20:row["char_end"] + 20])
                    print('doc_key', doc_key)
                    COUNTS['count_missed_type_2'] += 1

                elif expected == row.text:
                    temp = start_span + span_num_list[-2], end_span + span_num_list[-2], lookup[row["label"]]
                    aligned_entities[row["entity_id"]] = temp
                    COUNTS['count_aligned'] += 1


                else:
                    COUNTS['count_missed_type_3'] += 1
                    print('doc_key', doc_key)

            aligned = aligned_entities

        relations_sent = get_relations_in_sent(aligned, relations)

        num_aligned += len(relations_sent)

        # Append to result list
        toks = []
        for i in range(len(dict_final)):
            toks.append(dict_final[i][0])
        scierc_format["sentences"].append(toks)

        entities_to_scierc = [list(x) for x in aligned.values()]
        scierc_format["ner"].append(entities_to_scierc)

        scierc_format["relations"].append(relations_sent)
        scierc_format["doc_key"] = str(scierc_format["doc_key"])

    return scierc_format, len(relations) - num_aligned



COUNTS = Counter()
res = []
num_missed_tot = 0
for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):
    scierc_format, num_missed_doc = one_abstract(abstract, df_entities, df_relations)
    num_missed_tot += num_missed_doc
    res.append(scierc_format)

counts = pd.Series(COUNTS)
print(counts)
print('Number of missed relations:', num_missed_tot)


with open('train.json', "w") as f_out:
    for line in res:
        f_out.write(json.dumps(line))
        f_out.write('\n')

