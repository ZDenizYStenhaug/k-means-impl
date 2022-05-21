from sklearn.datasets import make_blobs, make_circles
from numpy import savetxt

DATASET = "CIRCLES"
N_SAMPLES = 500
N_CENTERS = 7

def generate_data():
    if DATASET == "BLOBS":
        generate_graph_blobs()
    elif DATASET == "CIRCLES":
        generate_graph_circular()
    else:
        print("Please enter a valid dataset.")


def generate_graph_blobs():
    points, labels_true = make_blobs(n_samples=N_SAMPLES, centers=N_CENTERS, n_features=2)
    save_to_file(points)


def generate_graph_circular():
    points, clusters = make_circles(n_samples=N_SAMPLES, noise=.05, factor=.5, random_state=0)
    save_to_file(points)


def save_to_file(points):
    savetxt('data.csv', points, delimiter=',')


if __name__ == '__main__':
    generate_data()
