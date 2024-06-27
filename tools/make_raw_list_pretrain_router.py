#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json


def key2domain(key):
    if key.find('BUS') != -1:
        domain = 1
    elif key.find('CAF') != -1:
        domain = 2
    elif key.find('PED') != -1:
        domain = 3
    elif key.find('STR') != -1:
        domain = 4
    else:
        domain = 0
    return domain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('output_file', help='output list file')
    args = parser.parse_args()

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    with open(args.output_file, 'w', encoding='utf8') as fout:
        count = 0
        for key in wav_table:
            assert key in wav_table
            wav = wav_table[key]
            line = dict(key=key, wav=wav, txt=key2domain(key))
            # line = dict(key=key, wav=wav, domain=key2domain(key))
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')

        print(count)
