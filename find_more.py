import os
import json
import requests
import numpy as np


def datamuse_v2(query, max_num, setting):
    words = []
    for info in setting:
        url = 'https://api.datamuse.com/%s=%s&max=%d' % (info, query, max_num)
        response = requests.get(url)
        results = response.text

        results = json.loads(results)

        words.extend([a['word'] for a in results])

    return words

if '__main__' == __name__:

    setting = ['words?ml', 'words?rel_spc', 'words?rel_gen', 'words?rel_par', 'words?rel_bga', 'words?rel_bgb', 'sug?s']

    output = datamuse_v2('dog', 100, setting)
    print(output)
