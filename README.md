# 🎬 Matrix Factorization for Movie Recommendations (MovieLens 100K)

This project implements a **collaborative filtering-based movie recommendation system** using **matrix factorization** with **gradient descent optimization**. It is trained and evaluated on the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), one of the most widely used datasets for evaluating recommender systems.

## 📌 Project Objective

- Predict user preferences for movies using latent factors.
- Implement and train matrix factorization from scratch using gradient descent.
- Explore how hyperparameters like number of latent factors (`k`), learning rate (`lr`), and regularization strength (`λ`) affect performance.
- Visualize convergence of training error and hyperparameter sensitivity.

---

## 📂 Project Structure

.
├── train.txt # Training data from MovieLens 100K (user, item, rating)
├── test.txt # Test data from MovieLens 100K
├── main.py # Main Python script with training, evaluation, visualization
├── P.npy # Learned user feature matrix
├── Q.npy # Learned item feature matrix
└── README.md # This file

yaml
Copy
Edit

---

## ⚙️ Methodology

The algorithm factorizes the user-item rating matrix `R` into two lower-rank matrices:

> **R ≈ P × Qᵀ**

Where:
- `P` is the user feature matrix (num_users × k)
- `Q` is the item feature matrix (num_items × k)
- `k` is the number of latent features (e.g., user preferences, item traits)

We optimize the objective function:

> **E = Σ(r_ui - p_u·q_i)² + λ(||p_u||² + ||q_i||²)**

using **stochastic gradient descent**.

---

## 🚀 How to Run

### ✅ Requirements

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install numpy matplotlib
▶️ Run the Project
bash
Copy
Edit
python main.py
This will:

Train the model on train.txt

Evaluate RMSE on both training and test sets

Save the P.npy and Q.npy matrices

Plot:

Error vs Iterations

Error vs Latent Factor k

Error vs Learning Rate

Error vs Regularization

📈 Visualizations
The code generates the following plots:

Error vs. Iterations: See how the objective function converges during training.

Error vs. Latent Factors (k): Analyze model performance for different dimensionalities.

Error vs. Learning Rate: Explore stability and convergence speed.

Error vs. Regularization Strength: Examine how overfitting is controlled.

🔢 Example Output (Console)
bash
Copy
Edit
Iteration 1/5 - Error: 92234.3125
Iteration 2/5 - Error: 84217.1953
...
Train RMSE: 0.9453
Test RMSE: 0.9812
🔧 Hyperparameter Tuning
You can easily tune:

k (latent factors): Try values from 10 to 50.

learning_rate: Start with 0.005 and increase gradually.

regularization: Controls overfitting (try 0.1 to 1.0).

num_iterations: Training epochs.

These are set at the top of the script.

🧪 Evaluation Metric
We use Root Mean Squared Error (RMSE) to evaluate prediction performance:

RMSE = sqrt(Σ(predicted - actual)² / N)

Computed separately for both training and test data.

🛠️ Customization
Change dataset: Replace train.txt and test.txt with other MovieLens splits.

Adjust training logic: Modify gradient_descent() in main.py.

Add additional plots or logging for deeper insights.

🤝 Contributing
Pull requests, bug reports, and ideas are welcome! If you find a way to improve convergence, handle cold-starts, or integrate implicit feedback — feel free to contribute.

📄 License
This project is licensed under the MIT License. See LICENSE for details.

📚 References
GroupLens MovieLens Dataset: https://grouplens.org/datasets/movielens/100k/

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer.
