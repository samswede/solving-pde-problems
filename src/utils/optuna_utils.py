import optuna
import multiprocessing
import time
import mlflow

def objective(trial, config, run_model, log_params):
    solver = trial.suggest_categorical("solver", list(config["solvers"].keys()))
    method = trial.suggest_categorical("method", config["solvers"][solver]["methods"])
    timeout = config["timeout"]

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue, config, solver, method, timeout, run_model, log_params))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        queue.join_thread()
        return timeout  # Return timeout value if timeout occurs

    result = queue.get()
    queue.close()
    queue.join_thread()
    return result  # Return the execution time if successful

def target(queue, config, solver, method, timeout, run_model, log_params):
    execution_time = run_with_timeout(config, solver, method, timeout, run_model, log_params)
    queue.put(execution_time)

def run_with_timeout(config, solver, method, timeout, run_model, log_params):
    config["solver_params"]["solver"] = solver
    config["solver_params"]["method"] = method
    run_name = f"{config['run_name_prefix']}_{solver}_{method}_grid_{config['grid_size']}"
    parent_run_id = config.get("parent_run_id")  # Get the parent run ID from the config

    with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id):
        try:
            mlflow.set_tag("pde_package", config.get("pde_package", "default"))
            mlflow.log_param("solver", solver)
            mlflow.log_param("solver_method", method)
            mlflow.log_param("timeout", timeout)
            mlflow.set_tag("parent_run", config.get("parent_run", "default_parent_run"))
            mlflow.set_tag("child_run_index", method)
            log_params(config)  # Log all parameters
            start_time = time.time()
            run_model(config)
            end_time = time.time()
            return end_time - start_time
        finally:
            mlflow.end_run()  # Ensure the run is properly ended

def run_optuna_study(optuna_config, run_model, log_params, n_trials=10):
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, optuna_config, run_model, log_params), n_trials=n_trials)

    # Log the best solver and method found
    best_solver = study.best_params["solver"]
    best_method = study.best_params["method"]
    mlflow.log_param("best_solver", best_solver)
    mlflow.log_param("best_solver_method", best_method)

    # Run the simulation with the best solver and method
    # run_with_timeout(optuna_config, best_solver, best_method, timeout=optuna_config["timeout"], run_model=run_model, log_params=log_params)