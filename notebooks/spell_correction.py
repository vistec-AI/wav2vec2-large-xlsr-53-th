'''
Scripts by @tann9949 (github.com/tann9949)
'''

import argparse
import multiprocessing as mp
import re
from functools import partial
from typing import Dict, List

import pandas as pd
from pythainlp.tokenize import word_tokenize
            

def format_repeat(text: str) -> str:
    """Format Thai sentence containing ๆ
    
    Arguments
    ---------
    text: str
        text to be processed

    Return
    ------
    formatted_text: str
        formatted text that got repeated by ๆ
    """
    # check whether sentence start with ๆ
    text = text.replace(" ", "");
    if text[0] == "ๆ":
        raise ValueError(f"ๆ must not be at the start of sentence: {text}");

    tokenized_text: List[str] = word_tokenize(text);
    formatted_text: List[str] = [];

    for i, word in enumerate(tokenized_text):
        if "ๆ" in word:
            splitted_word: List[str] = [x for x in re.split("(ๆ)", word) if x != ""];
            if splitted_word[0] == "ๆ":
                # if splitted words are all ๆ
                last_word: str = tokenized_text[i-1];
                for c in word:
                    if c == "ๆ":
                        formatted_text.append(last_word);
                formatted_text.append(word.replace("ๆ", ""));
            else:
                current_word = splitted_word[0];
                for w in splitted_word:
                    if w != "ๆ":
                        current_word: str = w;
                        formatted_text.append(w);
                    else:
                        formatted_text.append(current_word);
        else:
            formatted_text.append(word)
    return "".join(formatted_text);


def correct_sentence(sentence: str, custom_dict: Dict[str, str] = {}) -> None:
    """Correct misspell sentence according to the following rule
    1. check whether แ is spelled by เ + เ
    2. check whether ำ is spelled by  ํ + า
    3. check whether tonal mark ( ่,  ้,  ๊,  ๋ ) is followed after vowel ( ั, ำ, ุ, ู )
    and save it in output file

    Arguments
    ---------
    sentence: str
        Sentence to be corrected
    """
    tonal_marks: List[str] = ["่", "้", "๊", "๋"];
    vowel: List[str] = ["ั", "ุ", "ู", "ํ"];

    # replace custom dict
    for word, replace_word in custom_dict.items():
        if word in sentence:
            sentence = sentence.replace(word, replace_word);
            print(f"CUSTOM DICT: Replace `{word}` => `{replace_word}`");

    if "เเ" in sentence:
        sentence = sentence.replace("เเ", "แ");  # correct เ + เ -> แ
        print(f"Correct เ + เ => แ");
    if "ํา" in sentence:
        sentence = sentence.replace("ํา", "ำ");  # correct ํ + า -> ำ
        print(f"Correct ํ + า => ำ");
    if "ๆ" in sentence:
        sentence = format_repeat(sentence);
        print("ๆ Replaced")
    # correct #3
    corrected_sentence: str = sentence;
    for i in range(len(sentence) - 1):
        char: str = sentence[i];
        next_char: str = sentence[i+1];
        if char in tonal_marks and next_char in vowel:
                corrected_sentence: List[str] = list(corrected_sentence);
                corrected_sentence[i] = next_char;
                corrected_sentence[i+1] = char;
                corrected_sentence: str = str(corrected_sentence);
                print(f"Corrected `{char}` + `{next_char}` => `{next_char}` + `{char}`");
        if char == "ํ" and next_char in tonal_marks and sentence[i+2] == "า":
            corrected_sentence = corrected_sentence.replace(f"ํ{next_char}า", f"{next_char}ำ");
            print(f"Corrected `ํ` + `{next_char}` + `า` => `{next_char}` + `ำ`");
    
    return corrected_sentence