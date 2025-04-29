import sklearn.model_selection
from sklearn.datasets import load_diabetes, load_breast_cancer
from ucimlrepo import fetch_ucirepo, list_available_datasets
import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

def load_magic(input_folder='WEBSITE/DATA', output_phenotype='HET_2'):
    # You can download and extract the MAGIC dataset by
    # > wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
    # > tar -xvf BASIC_GWAS.tar.gz
    # Load data
    founder_names = ["Banco", "Bersee", "Brigadier", "Copain", "Cordiale", "Flamingo",
                     "Gladiator", "Holdfast", "Kloka", "MarisFundin", "Robigus", "Slejpner",
                     "Soissons", "Spark", "Steadfast", "Stetson"]

    # Genotype
    genotype = pd.read_csv(os.path.join(input_folder, 'MAGIC_IMPUTED_PRUNED/MAGIC_imputed.pruned.traw'),
                           sep='\t')
    genotype.set_index('SNP', inplace=True)
    genotype = genotype.iloc[:, 5:]
    colnames = genotype.keys()
    new_colnames = [c.split('_')[0] for c in colnames]
    genotype.rename(columns={c: new_c for c, new_c in zip(colnames, new_colnames)}, inplace=True)
    genotype = genotype.transpose()

    # Phenotype
    phenotype = pd.read_csv(os.path.join(input_folder, 'PHENOTYPES/NDM_phenotypes.tsv'), sep='\t')
    phenotype.set_index('line_name', inplace=True)
    phenotype.drop(founder_names, inplace=True)
    del phenotype['line_code']

    # Make genotype have the same index as phenotype
    genotype = genotype.reindex(phenotype.index, )

    # Replace NAs
    genotype = genotype.fillna(genotype.mean(axis=0))
    phenotype = phenotype.fillna(phenotype.mean(axis=0))

    # Formulate problem
    X = np.array(genotype.values)
    y = np.array(phenotype[output_phenotype].values)

    return X, y

def normalize(X_train, X_test, y_train, y_test):
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return X_train, X_test, y_train, y_test

# DATASET REGRESSION
def magic():
    X, y = load_magic(input_folder='../WEBSITE/DATA')
    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    X_train = X_train
    X_test = X_test
    return normalize(X_train, X_test, y_train, y_test)

def diabetes():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def wine():
    dset = fetch_ucirepo(name="Wine Quality")
    X = dset.data.features.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def abalone():
    dset = fetch_ucirepo(name="Abalone")
    F = dset.data.features
    F = F.assign(Sex=(F['Sex'] == 'M').values.astype(float))
    X = F.values
    y = dset.data.targets.values.flatten()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)



def polution():
    dset = fetch_openml(data_id=542)
    X = dset['data'].values
    y = dset['target'].values

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def diamonds():
    dset = fetch_openml(data_id=42225)
    df = dset['data']
    # Enocde Data
    cat_col = ['cut', 'clarity', 'color']
    le = LabelEncoder()
    for col in cat_col:
        df = df.assign(**{col: le.fit_transform(df[col])})

    X = df.values
    y = dset['target'].values

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)


def house_sales():
    dset = fetch_openml(data_id=42731)
    X = dset['data'].drop(columns='zipcode').values
    y = dset['target'].values

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)


def us_crime():
    dset = fetch_openml(data_id=42730)
    X = dset['data'].fillna(method='backfill').fillna(method='pad').values
    y = dset['target'].values

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return normalize(X_train, X_test, y_train, y_test)

def gaussian(n_train, n_test, n_params, seed=1, noise_std=0.1):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_params)
    X_test = rng.randn(n_test, n_params)
    beta = 1 / np.sqrt(n_params) * rng.randn(n_params)
    y_train = X_train @ beta + noise_std * rng.randn(n_train)
    y_test = X_test @ beta + noise_std * rng.randn(n_test)
    return X_train, X_test, y_train, y_test

def sparse_gaussian(n_train, n_test, n_params, non_zeros=10, seed=1, noise_std=0.1, parameter_norm=1):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_params)
    X_test = rng.randn(n_test, n_params)
    beta = np.zeros(n_params)
    ind = rng.choice(n_params, non_zeros, replace=False)
    beta[ind] = parameter_norm / np.sqrt(non_zeros) * rng.randn(non_zeros)
    y_train = X_train @ beta + noise_std * rng.randn(n_train)
    y_test = X_test @ beta + noise_std * rng.randn(n_test)
    return X_train, X_test, y_train, y_test


def generate_random_ortogonal(p, d, rng):
    """Generate random W with shape (p, d) such that `W.T W = p / d I_d`."""
    aux = rng.randn(p, d)
    q, r = np.linalg.qr(aux, mode='reduced')
    return q
def latent_features(n_train, n_test, n_params, n_latent=1, seed=1, noise_std=0.1, parameter_norm=1):
    rng = np.random.RandomState(seed)
    factor = np.sqrt(n_params / n_latent)
    theta = parameter_norm / np.sqrt(n_latent) * rng.randn(n_latent)
    w = factor * generate_random_ortogonal(n_params, n_latent, rng)
    z = rng.randn(n_train+n_test, n_latent)
    u = rng.randn(n_train+n_test, n_params)
    e = rng.randn(n_train+n_test)
    y = z @ theta + noise_std * e
    X = z @ w.T + u
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test)
    return X_train, X_test, y_train, y_test

def gaussian_classification(n_train, n_test, n_params, seed=1, noise_std=0.1, parameter_norm=1):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_params)
    X_test = rng.randn(n_test, n_params)
    beta = 1 / np.sqrt(n_params) * rng.randn(n_params)
    y_train = (np.sign(X_train @ beta + noise_std * rng.randn(n_train)) + 1) / 2
    y_test = (np.sign(X_test @ beta + noise_std * rng.randn(n_test)) + 1) / 2
    return X_train, X_test, y_train.astype(int), y_test.astype(int)

# Classification datasets (Y=0 or 1)
def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    X -= np.mean(X, axis=0)
    X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
    y = np.asarray(y, dtype=np.float64)
    return X[:400], X[400:], y[:400], y[400:]

def MNIST():
    X, y = fetch_openml('mnist_784', return_X_y=True)
    X = X.values.astype(np.float64)
    X /= 255  # Normalize between [0, 1]
    y = np.asarray(y == '9', dtype=np.float64)
    return X[:60000], X[60000:], y[:60000], y[60000:]

def MAGIC_C():
    X, y = load_magic(input_folder='../WEBSITE/DATA', output_phenotype='SH_1')
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_test, y_train, y_test

def heart_failure():
    dset = fetch_openml(data_id=43008)
    X = dset['data'].values[:, :-1]
    y = dset['data'].values[:, -1].astype(int)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def blood_transf():
    dset = fetch_openml(data_id=1464)
    X = dset['data'].values.astype(float)
    y = (dset['target'].values == '2').astype(int)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test
def iris():
    X, y_ = fetch_openml(data_id=41078, return_X_y=True)
    X = X.values
    y = np.asarray(y_ == 'Iris-setosa', dtype=np.float64)
    X -= np.mean(X, axis=0)
    X /= X.max(axis=0)  # Normalize each feature to be in [-1, 1], so adversarial training is fair
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=50, random_state=0)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gaussian_classification(100, 10, 100)
