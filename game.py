import itertools
import random
from fractions import Fraction
    
def create_random_hand():
    cards = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                cards.append(card(a,b,c,d))
                for d in range(3):
                    cards.append(card(a,b,c,d))

    random.shuffle(cards)
    return cards[0:cards_per_board]


def check_board(cards):
    valid_sets_shape = set()
    valid_sets_color = set()
    valid_sets_numb = set()
    valid_sets_inter = set()

    #for shape
    for comb in itertools.combinations(cards,3):
        # check for all different
        if comb[0].shape != comb[1].shape and comb[1].shape != comb[2].shape and comb[2].shape != comb[0].shape:
            valid_sets_shape.add(comb)
        # check for all same
        if comb[0].shape == comb[1].shape and comb[1].shape == comb[2].shape:
            valid_sets_shape.add(comb)

    #for color
    for comb in valid_sets_shape:
        # check for all different
        if comb[0].color != comb[1].color and comb[1].color != comb[2].color and comb[2].color != comb[0].color:
            valid_sets_color.add(comb)
        # check for all same
        if comb[0].color == comb[1].color and comb[1].color == comb[2].color:
            valid_sets_color.add(comb)

    #for number
    for comb in valid_sets_color:
        # check for all different
        if comb[0].numb != comb[1].numb and comb[1].numb != comb[2].numb and comb[2].numb != comb[0].numb:
            valid_sets_numb.add(comb)
        # check for all same
        if comb[0].numb == comb[1].numb and comb[1].numb == comb[2].numb:
            valid_sets_numb.add(comb)

    ##for inter
    #for comb in valid_sets_numb:
        # check for all different
        if comb[0].inter != comb[1].inter and comb[1].inter != comb[2].inter and comb[2].inter != comb[0].inter:
            valid_sets_inter.add(comb)
        # check for all same
        if comb[0].inter == comb[1].inter and comb[1].inter == comb[2].inter:
            valid_sets_inter.add(comb)

    valid_sets = set()
    for comb in valid_sets_color:
        if comb in valid_sets_shape and comb in valid_sets_numb and comb in valid_sets_inter:
            valid_sets.add(comb)

    print_set = False
    if print_set:
        for vset in valid_sets:
            print("-")
            numbs = [c.numb for c in vset]
            print("n: ",numbs)
            shapes = [c.shape for c in vset]
            print("s: ",shapes)
            inters = [c.inter for c in vset]
            print("i: ",inters)
            colors = [c.color for c in vset]
            print("c: ",colors)
            print("-")
    return valid_sets
