import csv


def read_data(file):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    csv_file = csv.reader(open(file, 'r'))
    for index, row in enumerate(csv_file):
        if index < 18000:
            x_train.append(row[0:10])
            y_train.append(row[10:13])
        else:
            x_test.append(row[0:10])
            y_test.append(row[10:13])
    print("Read End! x_train: ", len(x_train), " x_test: ", len(x_test))
    # return GtList, ImageList
    return x_train, y_train, x_test, y_test
