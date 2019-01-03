import random
import time

VOCABULARY = [
    "cancel",
    "computer",
    "yes",
    "no",
    "ok",
]

def main():
    listOfWords = random_list_of_words()
    print_words(listOfWords)

def random_list_of_words():
    wordNum = 100 / len(VOCABULARY)
    listOfWords = list()
    for word in VOCABULARY:
        listOfWords += [word] * wordNum
    random.shuffle(listOfWords)
    return listOfWords

def print_words(words):
    counter = 0
    wait_for(3)
    for word in words:
        print("{} {}".format(counter, word))
        counter += 1
        pause = random.uniform(4, 8)
        wait_for(pause)

def wait_for(seconds):
    time.sleep(seconds)

if __name__ == "__main__":
    main()
