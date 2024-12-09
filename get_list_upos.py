from os import PathLike
from typing import Literal
import conllu

Lemma = str
Upos = str

def parse_conllu(file_path: PathLike, mode: Literal["form", "lemma"] = "form") -> list[list[tuple[Lemma, Upos]]]:
    final_list: list[list[tuple[Lemma, Upos]]] = []

    data_file = open(file_path, "r", encoding="utf-8")
    for tokenlist in conllu.parse_incr(data_file):
        current_list: list[tuple[Lemma, Upos]] = []
        for token in tokenlist:
            current_list.append((token[mode], token["upos"]))
        final_list.append(current_list)

    return final_list


if __name__ == "__main__":
    # This file should not be used as __main__, it is just for testing purposes
    print(parse_conllu("en_lines-ud-dev.conllu"))
