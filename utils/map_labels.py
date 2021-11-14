def mapStrToInt(key):
    lookup = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return lookup[key]

def mapIntToStr(index):
    lookup = ["contradiction", "entailment", "neutral"]
    return lookup[index]