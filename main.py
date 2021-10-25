from scipy.stats import rankdata
from numpy import array, argsort, sqrt, rint

FILENAME_IN = 'in.txt'
FILENAME_OUT = 'out.txt'


def read_item(line):
    items = line.split(' ')
    return float(items[0]), float(items[1])


def read_data():
    try:
        with open(FILENAME_IN, "r") as f:
            x, y = zip(*[read_item(line) for line in f.readlines()])
        return x, y
    except Exception as e:
        print('Incorrect data format: ', e)
        exit(1)


if __name__ == '__main__':
    x, y = read_data()
    idx = argsort(x)
    x, y = array(x)[idx], array(y)[idx]
    ranks = rankdata(-y)
    n = len(x)

    if n < 9:
        print(f'Insufficient number of samples: {n} < 9')
        exit(1)

    p = int(n / 3)
    diff = sum(ranks[:p]) - sum(ranks[-p:])
    std = (n + 0.5) * sqrt(p / 6.)
    con = diff / p / (n - p)
    with open(FILENAME_OUT, "w") as f:
        f.write(f'{int(rint(diff))} {int(rint(std))} {round(con, 2)}')
