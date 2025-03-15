from collections import defaultdict

from datasets import load_dataset

xsum_ds = load_dataset("EdinburghNLP/xsum")
cnndm_ds = load_dataset("abisee/cnn_dailymail", "3.0.0")


def visualize():
    # List of lengths
    xsum_doc = defaultdict(list)
    xsum_sum = defaultdict(list)
    xsum_len = defaultdict(float)

    xsum_docavg = defaultdict(float)
    xsum_sumavg = defaultdict(float)
    for key in xsum_ds:
        for item in xsum_ds[key]:
            xsum_len[key] += 1.
            xsum_doc[key].append(len(item["document"]))
            xsum_sum[key].append(len(item["summary"]))

        xsum_docavg[key] = sum(xsum_doc[key]) / xsum_len[key]
        xsum_sumavg[key] = sum(xsum_sum[key]) / xsum_len[key]

    print("XSUM Length: " + str(xsum_len))
    print("XSUM Doc Length: " + str(xsum_docavg))
    print("XSUM Summary Length: " + str(xsum_sumavg))

    cnn_doc = defaultdict(list)
    cnn_sum = defaultdict(list)
    cnn_len = defaultdict(float)

    cnn_docavg = defaultdict(float)
    cnn_sumavg = defaultdict(float)
    for key in cnndm_ds:
        for item in cnndm_ds[key]:
            cnn_len[key] += 1.
            cnn_doc[key].append(len(item["article"]))
            cnn_sum[key].append(len(item["highlights"]))

        cnn_docavg[key] = sum(cnn_doc[key]) / cnn_len[key]
        cnn_sumavg[key] = sum(cnn_sum[key]) / cnn_len[key]

    print("cnn Length: " + str(cnn_len))
    print("cnn Doc Length: " + str(cnn_docavg))
    print("cnn Summary Length: " + str(cnn_sumavg))

    import matplotlib.pyplot as plt
    plt.hist(xsum_sum["train"], bins=[0, 64, 128, 192, 256, 320, 384, 448, 512])
    plt.xlim(xmin=0.0, xmax=512)
    plt.show()


if __name__ == '__main__':
    visualize()
