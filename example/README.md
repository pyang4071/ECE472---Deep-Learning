# Example Project

This project demonstrates a simple linear regression model built using JAX and Flax. It serves as a working example of the project template.

## 🚀 Running the Example

To run the example project, execute the following command from the root of the repository:

```bash
just run example
```

This will train a linear model on synthetic data and output the results.

## 🔬 What it Does

The script performs the following steps:

1.  **Generates Synthetic Data**: It creates a dataset from a known linear model with some added noise. The parameters of this "true" model are defined in `src/example/__init__.py`.
2.  **Initializes a Model**: A `NNXLinearModel` is created with random initial weights.
3.  **Trains the Model**: The model is trained using the Adam optimizer to fit the synthetic data.
4.  **Compares Results**: After training, the script compares the learned model parameters with the parameters of the original data-generating model.
5.  **Plots the Fit**: If the model has only one feature, it generates a plot (`fit.pdf`) showing the data points and the learned regression line. The plot is saved in the `artifacts/` directory.

## ⚙️ Configuration

You can customize the behavior of the example by editing `src/example/config.toml`. Key settings include:

-   `data.num_features`: The number of features in the synthetic data.
-   `data.num_samples`: The number of data points to generate.
-   `training.num_iters`: The number of training iterations.
-   `training.learning_rate`: The learning rate for the Adam optimizer.

### 🪵 Log Levels

The project uses `structlog` for logging. To see detailed `DEBUG` messages, set the `LOG_LEVEL` environment variable:

```bash
LOG_LEVEL=DEBUG just run example
```

This is useful for observing the model parameters at different stages of the process.
