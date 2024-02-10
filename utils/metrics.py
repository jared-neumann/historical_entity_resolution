def jaccard_similarity(name1, name2):
    name1_chars = set(name1)
    name2_chars = set(name2)
    intersection = name1_chars.intersection(name2_chars)
    if len(intersection) == 0:
        return 0
    union = name1_chars.union(name2_chars)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

