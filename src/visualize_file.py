import argparse
import json


def visualize(file):
    sum_len = []
    all_cnt = 0.
    for line in open(file, "r").readlines():
        item = json.loads(line)
        if "summary" in item:
            summary = item["summary"]
        elif "highlights" in item:
            summary = item["highlights"]
        else:
            continue
        sum_len.append(len(summary))
        all_cnt += 1.

    avg = sum(sum_len) / all_cnt
    print(avg)
    import matplotlib.pyplot as plt
    plt.hist(sum_len, bins=[0, 64, 128, 192, 256, 320, 384, 448, 512])
    plt.xlim(xmin=0.0, xmax=512)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    # run(args.path)
    visualize(args.path)
