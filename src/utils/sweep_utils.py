import multiprocessing
import time
import mlflow

def run_with_timeout(config, solver, method, timeout, run_model, log_params):
    config["solver_params"]["solver"] = solver
    config["solver_params"]["method"] = method
    run_name = f"{config['run_name_prefix']}_{solver}_{method}_grid_{config['grid_size']}"
    parent_run_id = config.get("parent_run_id")

    with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id):
        try:
            mlflow.set_tag("pde_package", config.get("pde_package", "default"))
            mlflow.set_tag("sweep", "true")
            mlflow.log_param("solver", solver)
            mlflow.log_param("solver_method", method)
            mlflow.log_param("timeout", timeout)
            mlflow.set_tag("parent_run", config.get("parent_run", "default_parent_run"))
            mlflow.set_tag("child_run_index", method)
            log_params(config)
            start_time = time.time()
            run_model(config)
            end_time = time.time()
            return end_time - start_time
        finally:
            mlflow.end_run()

def target(queue, config, solver, method, timeout, run_model, log_params):
    execution_time = run_with_timeout(config, solver, method, timeout, run_model, log_params)
    queue.put(execution_time)

def binary_search_grid_size(config, solver, method, timeout, run_model, log_params, tolerance=0.05, min_grid_size=10, max_grid_size=1000):
    low, high = min_grid_size, max_grid_size
    best_grid_size = None
    best_execution_time = None

    print(f"Starting binary search for solver: {solver}, method: {method}")

    while low <= high:
        mid = (low + high) // 2
        config["grid_size"] = mid

        print(f"Testing grid_size: {mid}")

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(queue, config, solver, method, timeout, run_model, log_params))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            queue.close()
            queue.join_thread()
            print(f"Timeout reached for grid_size: {mid}, reducing grid_size")
            high = mid - 1
        else:
            execution_time = queue.get()
            queue.close()
            queue.join_thread()

            print(f"Execution time for grid_size {mid}: {execution_time} seconds")

            if abs(execution_time - timeout) <= timeout * tolerance:
                best_grid_size = mid
                best_execution_time = execution_time
                print(f"Found suitable grid_size: {mid} with execution time: {execution_time} seconds")
                break
            elif execution_time < timeout:
                low = mid + 1
            else:
                high = mid - 1

    return best_grid_size, best_execution_time

def parameter_sweep(optuna_config, run_model, log_params, timeout, min_grid_size=10, max_grid_size=1000):
    for solver, methods in optuna_config["solvers"].items():
        for method in methods["methods"]:
            print(f"Starting parameter sweep for solver: {solver}, method: {method}")
            best_grid_size, best_execution_time = binary_search_grid_size(
                optuna_config, solver, method, timeout, run_model, log_params, min_grid_size=min_grid_size, max_grid_size=max_grid_size
            )
            if best_grid_size and best_execution_time:
                mlflow.log_params({
                    "solver": solver,
                    "method": method,
                    "grid_size": best_grid_size
                })
                mlflow.log_metric("execution_time", best_execution_time)
                print(f"Logged results for solver: {solver}, method: {method}, grid_size: {best_grid_size}, execution_time: {best_execution_time} seconds")
            else:
                print(f"No suitable grid_size found for solver: {solver}, method: {method}")