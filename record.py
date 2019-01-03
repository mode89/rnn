import random

VOCABULARY = [
    "cancel",
    "computer",
    "yes",
    "no",
    "ok",
]

def main():
    listOfWords = random_list_of_words()

def random_list_of_words():
    wordNum = 100 / len(VOCABULARY)
    listOfWords = list()
    for word in VOCABULARY:
        listOfWords += [word] * wordNum
    random.shuffle(listOfWords)
    return listOfWords

if __name__ == "__main__":
    main()
