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
    return list(filter(lambda t: is_valid_set(t), it.combinations(card_ids, 3)))

def is_valid_set(card_triple):
    """
    Idea: Look at the sum of every property (i.e. color), if they are all multiples of 3, we have a valid set.
    Explanation:
    for valid combinations
    - if they are all different: 0+1+2=3 this is a multiple of 3
    - if they are all the same: 3*x (0+0+0 or 1+1+1 or 2+2+2) this is also a multiple of 3
    for invalid combinations the sum is never a multiple of 3
    """
    card_0 = cards[card_triple[0]]
    card_1 = cards[card_triple[1]]
    card_2 = cards[card_triple[2]]
    return all([sum(prop)%3==0 for prop in zip(card_0, card_1, card_2)])