from perceptron import Perceptron
import numpy as np
import pandas as pd

def main():
    # load data into shuffle matrix
    df = pd.read_csv("data/diabetes.csv")
    df = np.array(df)
    np.random.shuffle(df)

    # split data into labels and design matrix
    design = df[:, :-1].T
    labels = df[:, -1]
    labels = labels.reshape((labels.shape[0],1)).T # keep dimensionality to 2
    _, m_tot = design.shape

    # split into test and training data
    frac_test = .6
    split_idx = int(frac_test*m_tot)
    train_design = design[:, :split_idx]
    train_labels = labels[:, :split_idx]
    test_design = design[:, split_idx:]
    test_labels = labels[:, split_idx:]

    # fit perceptron
    perc = Perceptron()
    perc.fit(X=train_design, Y=train_labels, alpha=1e-4, lambd=0.1, epochs=100_000)
    
    # get model accuracies
    test_acc = perc.acc(X=test_design, Y=test_labels)
    train_acc = perc.acc(X=train_design, Y=train_labels)
    print("Test set accuracy: %.5f" % test_acc)
    print("Training set accuracy: %.5f" % train_acc)

if __name__ == "__main__":
    main()