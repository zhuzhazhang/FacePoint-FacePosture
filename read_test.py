import csv


def read_test(file):
    x_test = []
    csv_file = csv.reader(open(file, 'r'))
    for index, row in enumerate(csv_file):
        x_test.append(row[0:10])
    print("Read End! x_test: ", len(x_test))
    # return GtList, ImageList
    return x_test
