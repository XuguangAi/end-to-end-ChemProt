task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'chemprot': ['CHEMICAL', 'GENE'],
    'chemprot_5': ['CHEMICAL', 'GENE'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'chemprot': ['INHIBITOR', 'REGULATOR', 'DIRECT-REGULATOR', 'SUBSTRATE', 'INDIRECT-DOWNREGULATOR', 'INDIRECT-UPREGULATOR', 'ACTIVATOR', 'ANTAGONIST', 
                 'INDIRECT-REGULATOR', 'NOT', 'PART-OF', 'PRODUCT-OF', 'AGONIST', 'DOWNREGULATOR', 'UPREGULATOR', 'COFACTOR', 'MODULATOR-ACTIVATOR', 
                'SUBSTRATE_PRODUCT-OF', 'AGONIST-ACTIVATOR', 'MODULATOR', 'AGONIST-INHIBITOR', 'MODULATOR-INHIBITOR', 'UNDEFINED'],
    'chemprot_5': ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9']
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
