import itertools as it

cards = dict()
card_id = 0
for number in range(3):
    for color in range(3):
        for form in range(3):
            for filling in range(3):
                cards[card_id] = (number, color, form, filling)
                card_id += 1

def find_sets(card_ids):
    valid_sets = list()
    for triple in it.combinations(card_ids, 3):
        if is_valid_set(triple):
            valid_sets.append(triple)
    return valid_sets

def is_valid_set(triple):
    card_0 = cards[triple[0]]
    card_1 = cards[triple[1]]
    card_2 = cards[triple[2]]

    for idx in range(len(card_0)):
        diff_values = set([card_0[idx], card_1[idx], card_2[idx]])
        if len(diff_values) == 2:
            return False
    return True