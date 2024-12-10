import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import accuracy_score,root_mean_squared_error as rmse, mean_squared_error as mse
from ucimlrepo import fetch_ucirepo
from typing import Union,Tuple,List
from matplotlib import pyplot as plt
import seaborn as sns
census_income = fetch_ucirepo(id=20) 
encoded_copy=census_income.copy()
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']




encoded_copy = census_income.data.features.copy()
encoded_copy['income'] = census_income.data.targets
encoded_copy['income'] = encoded_copy['income'].apply(lambda x: 0 if x == '<=50K' else 1)

encoder = sklearn.preprocessing.LabelEncoder()
for feature in categorical_features:
    encoded_copy[feature] = encoder.fit_transform(encoded_copy[feature].astype(str))


X = encoded_copy.drop(columns='income').to_numpy()
y = encoder.fit_transform(encoded_copy['income']).reshape(-1, 1)

#Preprocess data


X_trn, X_tst, y_trn, y_tst = sklearn.model_selection.train_test_split(X, y, train_size=.8, random_state=42)
X_trn, X_vld, y_trn, y_vld = sklearn.model_selection.train_test_split(X_trn, y_trn, train_size=.8, random_state=42)

print(np.unique(y_trn))




plt.figure(figsize=(12, 6))
sns.kdeplot(data=encoded_copy[encoded_copy['income'] == 0], x='age', label='<=50K', fill=True, alpha=0.4)
sns.kdeplot(data=encoded_copy[encoded_copy['income'] == 1], x='age', label='>50K', fill=True, alpha=0.4)
plt.title('Age Distribution by Income Level')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Income Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(data=encoded_copy[encoded_copy['income'] == 0], x='education', label='<=50K', fill=True, alpha=0.4)
sns.kdeplot(data=encoded_copy[encoded_copy['income'] == 1], x='education', label='>50K', fill=True, alpha=0.4)
plt.title('Education Distribution by Income Level')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Income Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

def plot_feature_distributions(X_original, X_scaled, feature_names):
    num_features_to_plot = min(5, len(feature_names))  
    fig, axs = plt.subplots(2, num_features_to_plot, figsize=(15, 8))

    for i in range(num_features_to_plot):
        axs[0, i].hist(X_original[:, i], bins=30, alpha=0.7, color='blue')
        axs[0, i].set_title(f"Original: {feature_names[i]}")
        axs[0, i].grid()

        axs[1, i].hist(X_scaled[:, i], bins=30, alpha=0.7, color='orange')
        axs[1, i].set_title(f"Scaled: {feature_names[i]}")
        axs[1, i].grid()

    plt.tight_layout()
    plt.show()


feature_names = list(encoded_copy.drop(columns='income').columns) 
plot_feature_distributions(X_original=encoded_copy.drop(columns='income').to_numpy(), 
                           X_scaled=X_trn, 
                           feature_names=feature_names)


def plot_predictions(y_true,y_pred):
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Income")
    plt.ylabel("Predicted Income")
    plt.title("Actual vs Predicted Income")
    plt.grid()
    plt.show()
    


def compute_log_priors(y:np.array): 
    unique_classes, counts = np.unique(y, return_counts=True)
    N=y.shape[0]
    log_priors=np.log(counts/N)

      
    return log_priors


def compute_log_gaussian(x:np.array,    mu: Union[np.ndarray, float], sigma: Union[np.ndarray, float]):
    a=(-1/2)*np.log(np.square(sigma)*2*np.pi)
    b=0.5*np.square((x-mu)/sigma)
    return a-b

def compute_parameters(X:np.array,y:np.array):
    means = []
    stds = []
    n_features=X.shape[1]
    K =np.unique(y) 
    # TODO 5
    for i in K:
        X_i=X[y.ravel()==i]
        mean=np.mean(X_i,axis=0)
        means.append(mean)
        sigma=np.std(X_i,axis=0)
        stds.append(sigma)
    
    means = np.array(means)
    sigma = np.array(stds)
    print(sigma.shape)
    print(means.shape)
    return means,sigma

def compute_log_likelihood(X,means,stds,verbose=True):
    log_likelihoods = []
    for mean_k,std_k in zip(means,stds):
        ll_k= -0.5*np.log((std_k**2)*2*np.pi) - 0.5*np.square((X-mean_k)/std_k)
        log_likelihoods.append(np.sum(ll_k,axis=1))
    ll=np.column_stack(log_likelihoods)
    return ll

class GNB():
    def __init__(self):
        self.log_priors=None
        self.means = None
        self.stds = None
    def fit(self, X: np.ndarray, y: np.ndarray) -> object:

        self.log_priors=compute_log_priors(y)
        self.means,self.stds=compute_parameters(X,y)
        return self
    def predict(self,X:np.array):
        log_likelihoods = compute_log_likelihood(X, self.means, self.stds)
        posteriors=log_likelihoods+self.log_priors
        predictions=np.argmax(posteriors,axis=1)
        return predictions


gnb=GNB()
gnb.fit(X_trn,y_trn)

#Training
y_trn_hat=gnb.predict(X_trn)
y_trn_acc=accuracy_score(y_trn,y_trn_hat)
print(f"Training Accuracy: {y_trn_acc }")
print(f"RMSE:{rmse(y_trn,y_trn_hat)}")

#Validation

y_vld_hat=gnb.predict(X_vld)
y_vld_acc=accuracy_score(y_vld,y_vld_hat)
print(f"Validation Accuracy: {y_vld_acc}")
print(f"RMSE:{rmse(y_vld,y_vld_hat)}")

#Testing
y_tst_hat=gnb.predict(X_tst)
y_tst_acc=accuracy_score(y_tst,y_tst_hat)
print(f"test accuracy: {y_tst_acc}")
print(f"RMSE:{rmse(y_tst,y_tst_hat)}")




##Neural Network

scaler=sklearn.preprocessing.StandardScaler()
X_trn=scaler.fit_transform(X_trn,y=y)
X_vld=scaler.transform(X_vld)
X_tst=scaler.transform(X_tst)
y_trn = y_trn.reshape(-1, 1)
y_vld = y_vld.reshape(-1, 1)
y_tst = y_tst.reshape(-1, 1)

class Tanh:
    @staticmethod
    def activation(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z)**2


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_neurons: int, output_neurons: int, alpha: float, batch_size: int, epochs: int):
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.alpha = alpha  
        self.batch_size = batch_size
        self.epochs = epochs

        rng = np.random.RandomState(42)
        self.W1 = rng.uniform(-0.5, 0.5, (input_size, hidden_neurons))
        self.b1 = np.zeros((1, hidden_neurons))
        self.W2 = rng.uniform(-0.5, 0.5, (hidden_neurons, output_neurons))
        self.b2 = np.zeros((1, output_neurons))

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = Tanh.activation(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = Tanh.activation(Z2)  
        return Z1, A1, Z2, A2

    def backward(self, X: np.ndarray, y: np.ndarray, Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray):
        m = y.shape[0]  
        dZ2 = A2 - y 
        dW2 = A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * Tanh.derivative(Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1

    def fit(self, X: np.ndarray, y: np.ndarray):
        training_losses=[]
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                Z1, A1, Z2, A2 = self.forward(X_batch)

                self.backward(X_batch, y_batch, Z1, A1, Z2, A2)

            y_train_pred = self.predict(X)
            train_loss = mse(y, y_train_pred)
            training_losses.append(train_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss}")
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epochs), np.array(training_losses)**0.5, label='RMSE')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title("Training RMSE Over Epochs")
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, X: np.ndarray):
        _, _, _, A2 = self.forward(X)
        return A2   
nn = NeuralNetwork(
    input_size=X_trn.shape[1],
    hidden_neurons=30,
    output_neurons=1,
    alpha=0.01,
    batch_size=32,
    epochs=100
)

nn.fit(X_trn, y_trn)
y_pred_train = nn.predict(X_trn)
print("Train RMSE:", mse(y_trn, y_pred_train)**0.5)

y_pred_vld = nn.predict(X_vld)
print("Validation RMSE:", mse(y_vld, y_pred_vld)**0.5)

plot_predictions(y_vld,y_pred_vld)

y_pred_test = nn.predict(X_tst)
print("Test RMSE:", mse(y_tst, y_pred_test)**0.5)

plot_predictions(y_tst, y_pred_test)