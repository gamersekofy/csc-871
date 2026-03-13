// Misc setup
#set page(paper: "us-letter")

// Document metadata
#set document(
  title: "MLP for Regression",
  author: "Uzair Hamed Mohammed",
  date: auto,
)

#set par(justify: true)

// Title page
#align(center)[
  #title() \
  Uzair Hamed Mohammed \
  CSC 871, Spring 2026 \
  Due 03/12
]

#pagebreak()

// Establish header
#set page(
  header: align(center)[CSC 871/671 HW 4]
)

// Optional table of contents
#set page(numbering: "i.")
#counter(page).update(1)
#set heading(numbering: "1.")
#outline()

#pagebreak()

// Actual report
#set page(numbering: "1")
#counter(page).update(1)

= Introduction

In this assignment, we implement a Multi‑Layer Perceptron (MLP) for a regression task using PyTorch. The dataset consists of 1,000 points $(x, y)$ where the relationship between $x$ and $y$ is non‑linear. While a simple linear regression would underfit, an MLP with hidden layers and non‑linear activations can capture the underlying function. The goal is to build, train, and evaluate an MLP, and then explore the effect of different hyperparameters on the model’s performance.

The assignment requires:
- Normalizing the data using z‑score standardization.
- Constructing an MLP with three `torch.nn.Linear` layers (two hidden, one output).
- Using Mean Squared Error (MSE) as the loss function.
- Training the model and plotting the training loss over epochs.
- Plotting the final regression curve on the original data scale.
- Conducting a hyperparameter search to find a good combination.

This report documents the methodology, results, and conclusions from the experiments.

= Methodology

== Data Normalization

To improve training stability and convergence, the input $x$ and target $y$ were normalized using z‑score standardization:

$x_"norm" = (x - mu_x) / sigma_x$, $y_"norm" = (y - mu_y) / sigma_y$

The statistics computed from the training data are:

- $mu_x$ = 5.0647
- $sigma_x$ = 2.86835
- $mu_y$ = 9.7616
- $sigma_y$ = 2.7745

These values are later used to denormalize predictions when plotting the regression curve.

== MLP Architecture

The MLP consists of three linear layers:
- *Hidden layer 1*: input size 1 $->$ output size 50
- *Hidden layer 2*: input size 50 $->$ output size 25
- *Output layer*: input size 25 $->$ output size 1 (regression output)

After each hidden layer, a non‑linear activation function is applied (ReLU or Tanh). No activation is used on the output layer to allow the model to produce any continuous value.

The model is defined as:

```python
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden1=50, hidden2=25, output_dim=1, activation=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
```

== Training Details

All models were trained using the Mean Squared Error loss (`nn.MSELoss()`). Three different hyperparameter combinations were explored:

1. *Combo 1*: ReLU activation, Adam optimizer, learning rate = 0.01, batch size = 32.
2. *Combo 2*: Tanh activation, Adam optimizer, learning rate = 0.01, batch size = 32.
3. *Combo 3*: Tanh activation, Adam optimizer, learning rate = 0.001, batch size = 64.

Each model was trained for 200 epochs. The `DataLoader` shuffled the data every epoch. A reusable training function recorded the average loss per epoch.

== Hyperparameter Search Rationale

- *Activation functions*: ReLU is commonly used, but Tanh might be better suited if the data has both positive and negative values (as it is zero‑centered).
- *Learning rate*: Lower learning rates can allow more stable convergence, especially with larger batch sizes.
- *Batch size*: Larger batches provide less noisy gradient estimates but may require more epochs to converge. The combination of a lower learning rate and larger batch size was hypothesized to yield a better final fit.

= Results

== Training Loss Comparison

#figure(
  image("./resources/loss_comparison.png", width: 60%),
  caption: [
    Training loss (MSE) over 200 epochs for the three hyperparameter combinations.
  ]
) <loss-plot>

The loss curves (@loss-plot) show that *Combo 3* (Tanh, lr=0.001, batch size=64) achieves the lowest final loss, while *Combo 1* (ReLU) plateaus at a higher loss. *Combo 2* (Tanh with lr=0.01) initially drops quickly but then becomes unstable.

The slower learning rate in Combo 3 allows the optimizer to fine‑tune the weights more precisely, and the larger batch size reduces gradient variance, leading to a smoother and lower final loss.

== Regression Curves

After training, each model was evaluated on a dense set of $x$ values covering the original data range. The predictions were denormalized using $y_"pred" = y_"pred,norm" * sigma_y + mu_y$ and plotted against the original data.

#figure(
  image("./resources/regression_fits.png", width: 60%),
  caption: [
    Fitted regression curves for the three combinations overlaid on the original data.
  ]
) <regression-plot>

From @regression-plot, we observe:
- *Combo 1* (red) captures the general trend but fails to fit the peaks and troughs accurately.
- *Combo 2* (green) fits much better but shows slight oscillations, indicating some instability.
- *Combo 3* (blue) provides the smoothest and most accurate fit, closely following the data’s non‑linear pattern.

These qualitative observations align with the final loss values:

- Combo 1 final loss: 0.173082
- Combo 2 final loss: 0.109095
- Combo 3 final loss: 0.074849

= Conclusion

This assignment successfully demonstrated the use of an MLP for non‑linear regression. By comparing three hyperparameter combinations, we found that using *Tanh activation, a lower learning rate (0.001), and a larger batch size (64)* produced the best model, achieving a final MSE of 0.075 and a visually excellent fit to the data.

The experiments highlight the importance of tuning hyperparameters: a well‑chosen activation function and a careful balance between learning rate and batch size can significantly improve performance. The slower learning rate allowed more stable convergence, while the larger batch size provided less noisy gradient estimates, together yielding a model that generalizes well to the underlying function.
