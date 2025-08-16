import numpy as np
from numpy.linalg import matrix_power, inv
from datetime import datetime
import matplotlib.pyplot as plt


matricies = {
    "AA": {"matrix": np.matrix([[1, 0.5, 0], [0, 0.5, 1], [0, 0, 0]]), "P": np.matrix([[1, 1, -1], [-2, 0, 1], [1, 0, 0]])},
    "Aa": {"matrix": np.matrix([[0.5, 0.25, 0], [0.5, 0.5, 0.5], [0, 0.25, 0.5]]), "P": np.matrix([[1, 1, -1], [-2, 2, 0], [1, 1, 1]])},
    "aa": {"matrix": np.matrix([[0, 0, 0], [1, 0.5, 0], [0, 0.5, 1]]), "P": np.matrix([[1, 0, 0], [-2, 0, -1], [1, 1, 1]])}
}
genotypes_probability = np.matrix([[1/3],[1/3],[1/3]])
NGenerations = 100
parents = "aa"
matrix = matricies[f"{parents}"]
Evals = [0, 1, 0.5]

def diag_predict(NGenerations, matrix):
    D = np.diag(Evals)
    D = matrix_power(D, NGenerations)

    PInv = inv(matrix['P'])

    start_time = datetime.now()
    pdp = matrix['P'].dot(D.dot(PInv))
    end_time = datetime.now()
    result = pdp.dot(genotypes_probability).round(decimals=3)
    print(result)
    print('Diagonalization duration on', NGenerations, 'generations: {}'.format(end_time - start_time))

def predict(NGenerations, matrix):
    AA_points = []
    Aa_points = []
    aa_points = []

    start_time = datetime.now()
    M = matrix['matrix']

    for i in range(NGenerations):
        prediction = M.dot(genotypes_probability)
        prediction = (np.asarray(prediction)).flatten()
        AA_points.append(prediction[0]*100)
        Aa_points.append(prediction[1]*100)
        aa_points.append(prediction[2]*100)
        M = M.dot(matrix['matrix'])
        
    end_time = datetime.now()
    print('Multiplication duration on', NGenerations, 'generations: {}'.format(end_time - start_time))

    return AA_points, Aa_points, aa_points

def plotgraph(NGenerations, AA_points, Aa_points, aa_points, parents):
    print(parents)
    fig, ax = plt.subplots()
    x = np.array(range(1, NGenerations+1))
    y = np.array(AA_points)
    ax.plot(x,y, marker = "o", label = "AA offspring")

    x = np.array(range(1, NGenerations+1))
    y = np.array(Aa_points)
    ax.plot(x,y, marker = "o", label = "Aa offspring")

    x = np.array(range(1, NGenerations+1))
    y = np.array(aa_points)
    ax.plot(x,y,  marker = "o", label = "aa offspring")

    ax.set(xlabel='Generations', ylabel='Probability',
        title=f"{parents} Parents on {NGenerations} generations")
    ax.grid()
    ax.legend()
    fig.savefig(f"{parents} Parents on {NGenerations} generations")
    plt.show()

diag_predict(NGenerations, matrix)
AA_points, Aa_points, aa_points = predict(NGenerations, matrix)
plotgraph(NGenerations, AA_points, Aa_points, aa_points, parents)
