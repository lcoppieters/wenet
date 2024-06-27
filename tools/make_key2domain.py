import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('output_file', help='output list file')
    args = parser.parse_args()

wav_table = {}

with open(args.wav_file, 'r', encoding='utf8') as fin,  \
     open(args.output_file, 'w', encoding='utf8') as fout:
    for line in fin:
        arr = line.strip().split()
        assert len(arr) == 2
        key = arr[0]
        if key.find('BUS') != -1:
            domain = 1
        elif key.find('STR') != -1:
            domain = 2
        elif key.find('PED') != -1:
            domain = 3
        elif key.find('CAF') != -1:
            domain = 4
        else:
            domain = 0
        line = dict(key=key, domain=domain)
        json_line = json.dumps(line, ensure_ascii=False)
        fout.write(json_line + '\n')
