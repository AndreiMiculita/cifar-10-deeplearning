import csv
import matplotlib.pyplot as plt

with open("loss_over_time1.csv") as loss_over_time:
    reader = csv.reader(loss_over_time)
    lists = list(reader)
    x = range(len(lists[0])-3)

    for i in range(72, 84, 2):
        print("row", i)
        plist = [float(x) for x in lists[i][3:]]
        plt.plot(x, plist, label="VGG " + lists[i][2].split("(")[0])

    plist = [float(x) for x in lists[44][3:]]
    plt.plot(x, plist, label="ResNet " + lists[45][2].split("(")[0])
    plist = [float(x) for x in lists[40][3:]]
    plt.plot(x, plist, label="ResNet " + lists[41][2].split("(")[0])
    plt.xlabel("Epochs")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.show()
