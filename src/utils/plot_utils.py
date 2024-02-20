from matplotlib import pyplot as plt


def plot_seq_length(tokenized_sequences: list) -> None:
    sequence_lengths = [len(seq) for seq in tokenized_sequences]
    plt.hist(sequence_lengths, bins=20, edgecolor="black")
    plt.title("Histogram of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.show()
