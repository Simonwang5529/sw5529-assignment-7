from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
from scipy import stats
from scipy.stats import t
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + np.random.normal(mu, np.sqrt(sigma2), N)

        # Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts
    plt.figure()
    plt.hist(slopes, bins=30, alpha=0.7, label="Simulated Slopes")
    plt.hist(intercepts, bins=30, alpha=0.7, label="Simulated Intercepts")
    plt.axvline(slope, color="red", linestyle="dashed", linewidth=1, label="Observed Slope")
    plt.axvline(intercept, color="blue", linestyle="dashed", linewidth=1, label="Observed Intercept")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(1 for sim_slope in slopes if abs(sim_slope) >= abs(slope)) / S
    intercept_more_extreme = sum(1 for sim_intercept in intercepts if abs(sim_intercept) >= abs(intercept)) / S

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and store in session
        (X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme, slopes, intercepts) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Set all required session variables
        session["N"] = N
        session["S"] = S
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme

        # Render the index template with the generated plots
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # Get parameters from form or request (like `N`, `mu`, `sigma2`, etc.)
    N = int(request.form.get("N"))
    mu = float(request.form.get("mu"))
    sigma2 = float(request.form.get("sigma2"))
    beta0 = float(request.form.get("beta0"))
    beta1 = float(request.form.get("beta1"))
    S = int(request.form.get("S"))

    # Generate data and store it in the session
    X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme, slopes, intercepts = generate_data(
        N, mu, beta0, beta1, sigma2, S
    )

    # Set session variables
    session["N"] = N
    session["S"] = S
    session["mu"] = mu
    session["sigma2"] = sigma2
    session["beta0"] = beta0
    session["beta1"] = beta1
    session["slope"] = slope
    session["intercept"] = intercept
    session["slopes"] = slopes
    session["intercepts"] = intercepts
    session["slope_extreme"] = slope_extreme
    session["intercept_extreme"] = intercept_extreme

    # Render the template with generated data and plots
    return render_template(
        "index.html",
        plot1=plot1,
        plot2=plot2,
        slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Select appropriate dataset for the parameter being tested
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # test_type == "not_equal"
        p_value = 2 * min(
            np.mean(simulated_stats >= observed_stat),
            np.mean(simulated_stats <= observed_stat)
        )

    # Check for significant p-value and set a fun message
    fun_message = "Wow! This result is highly significant! 🎉" if p_value <= 0.0001 else None

    # Plotting
    plot_path = "static/plot3.png"
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, density=True, alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='--',
                label=f'Observed ({observed_stat:.3f})')
    plt.axvline(hypothesized_value, color='green', linestyle='--',
                label=f'Hypothesized ({hypothesized_value:.3f})')
    plt.title(f'Hypothesis Test Distribution (p-value = {p_value:.4f})')
    plt.xlabel('Statistic Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    # Return the results along with plots and messages
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )
@app.route("/confidence_interval", methods=["POST"])
@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Check if there's data available
    if 'N' not in session or 'slopes' not in session or 'intercepts' not in session:
        return render_template("index.html", error="Please generate data first")

    # Extract data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    # Determine which parameter to calculate the confidence interval for
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Select the appropriate data set based on the chosen parameter
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate the mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)  # ddof=1 provides an unbiased estimator

    # Calculate the critical value and the confidence interval
    alpha = 1 - confidence_level / 100
    t_crit = stats.t.ppf(1 - alpha/2, df=len(estimates)-1)
    ci_lower = mean_estimate - t_crit * std_estimate / np.sqrt(len(estimates))
    ci_upper = mean_estimate + t_crit * std_estimate / np.sqrt(len(estimates))
    includes_true = ci_lower <= true_param <= ci_upper

    # Create a plot for the confidence interval
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    y_jitter = np.zeros_like(estimates) + np.random.normal(0, 0.005, len(estimates))
    plt.scatter(estimates, y_jitter, color='gray', alpha=0.5, label='Simulated Estimates', s=30)
    ci_color = 'green' if includes_true else 'red'
    plt.hlines(0, ci_lower, ci_upper, colors=ci_color, linewidth=2, label=f'{confidence_level}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    plt.plot(mean_estimate, 0, 'o', color=ci_color, markersize=10, label='Mean Estimate')
    plt.axvline(true_param, color='blue', linestyle='--', label=f'True {parameter}: {true_param:.3f}')
    plt.title('Confidence Interval Plot')
    plt.xlabel('Parameter Value')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # Render the updated page with the new confidence interval plot
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )



if __name__ == "__main__":
    app.run(debug=True)
