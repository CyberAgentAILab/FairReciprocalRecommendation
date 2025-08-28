import json
import time
from market import Market
from naive import naive
from prod import prod
from iter_lp import iter_lp
from tu_matching import tu_matching
from alternate_fw import sw_maximize, nsw_maximize, alpha_sw_maximize
from alternate_fw_sinkhorn import sw_sinkhorn, nsw_sinkhorn

def experiment1(
        result_dir: str = ".",
        num_left: int = 10,
        num_right: int = 10,
        v_type: str = 'log',
        lambda_value: float = 0.8,
        testcases: int = 10,
        solver: str = "CLARABEL"
) -> None:
    """
    Conducts experiments in the setting of "5.1 Synthetic Data" in Tomita and Yokoyama [1], and "8.2 Synthetic Data Experiment I" in Tomita and Yokoyama (2025).

    Parameters
    ----------
    result_dir : str
        The directory where the results will be saved.
    num_left : int
        The number of left users.
    num_right : int
        The number of right users.
    v_type : str
        The type of examination functions ('log' or 'inv').
    lambda_value : float
        The lambda value for the preference generation.
    testcases : int
        The number of test cases to run.
    solver : str
        The CVXPY solver used in the IterLP, SW and NSW methods. "CLARABEL", "ECOS" or etc.
    """
    result = []
    for seed in range(testcases):
        print(f"\n========== Seed: {seed} ==========")
        m = Market(num_left=num_left, num_right=num_right, v_left_type=v_type, v_right_type=v_type)
        m.generate_preferences(pref_seed=seed, lambda_value=lambda_value)

        for method in ["Naive", "Prod", "IterLP", "TU", "SW", "NSW"]:
            start_time = time.time()
            if method == "Naive":
                left_rec = naive(m.pref_left_to_right)
                right_rec = naive(m.pref_right_to_left)
            elif method == "Prod":
                left_rec = prod(m.pref_left_to_right, m.pref_right_to_left)
                right_rec = prod(m.pref_right_to_left, m.pref_left_to_right)
            elif method == "IterLP":
                left_rec, right_rec = iter_lp(m.pref_left_to_right, m.pref_right_to_left, solver=solver)
            elif method == "TU":
                left_rec, right_rec = tu_matching(m.pref_left_to_right, m.pref_right_to_left, output=False)
            elif method == "SW":
                left_rec, right_rec = sw_maximize(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, solver=solver, output=False)
            elif method == "NSW":
                left_rec, right_rec = nsw_maximize(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, solver=solver, output=False)
            else:
                raise ValueError("Invalid Method")
            execution_time = time.time()-start_time
            
            match_prob = m.get_match_prob(left_rec, right_rec)
            envy = m.check_envy(left_rec, right_rec, match_prob)
            gini = m.compute_gini(match_prob)
            result.append(
                {
                    "NumLeft": num_left,
                    "NumRight": num_right,
                    "VType": v_type,
                    "Lambda": lambda_value,
                    "Seed": seed,
                    "Method": method,
                    "Match": match_prob.sum(),
                    "LeftEnvy": len(envy['left']),
                    "RightEnvy": len(envy['right']),
                    "LeftGini": gini[0],
                    "RightGini": gini[1],
                    "ExecutionTime": execution_time
                }
            )
            print(f"{method}: Match={result[-1]['Match']}, LeftEnvy={result[-1]['LeftEnvy']}, RightEnvy={result[-1]['RightEnvy']}, LeftGini={result[-1]['LeftGini']}, RightGini={result[-1]['RightGini']}, ExecutionTime={result[-1]['ExecutionTime']}")

    result_file_path = f'{result_dir}/experiment1_n{num_left}_m{num_right}_v{v_type}_lambda{lambda_value}_testcases{testcases}.json'

    with open(result_file_path, "w") as f:
        json.dump(result, f, indent=4)


def experiment2(
        result_dir: str = ".",
        num_left: int = 10,
        num_right: int = 10,
        v_type: str = 'log',
        lambda_value: float = 0.8,
        testcases: int = 10,
        solver: str = "CLARABEL"
) -> None:
    """
    Conducts experiments in the setting of "8.3 Synthetic Data Experiment II" in Tomita and Yokoyama (2025).

    Parameters
    ----------
    result_dir: str
        The directory to save the results.
    num_left: int
        The number of left users.
    num_right: int
        The number of right users.
    v_type: str
        The type of valuation functions ('log' or 'inv').
    lambda_value: float
        The lambda value for the preference generation.
    testcases: int
        The number of test cases to run.
    solver: str
        The CVXPY solver used in the alpha-SW methods. "CLARABEL", "ECOS" or etc.
    """
    result = []
    for seed in range(testcases):
        print(f"\n========== Seed: {seed} ==========")
        m = Market(num_left=num_left, num_right=num_right, v_left_type=v_type, v_right_type=v_type)
        m.generate_preferences(pref_seed=seed, lambda_value=lambda_value)

        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            start_time = time.time()
            left_rec, right_rec = alpha_sw_maximize(m.pref_left_to_right, m.pref_right_to_left, v_left=m.v_left, v_right=m.v_right, solver=solver, output=False, alpha=alpha)
            execution_time = time.time()-start_time

            match_prob = m.get_match_prob(left_rec, right_rec)
            envy = m.check_envy(left_rec, right_rec, match_prob)
            gini = m.compute_gini(match_prob)
            result.append(
                {
                    "NumLeft": num_left,
                    "NumRight": num_right,
                    "VType": v_type,
                    "Lambda": lambda_value,
                    "Seed": seed,
                    "Alpha": alpha,
                    "Match": match_prob.sum(),
                    "LeftEnvy": len(envy['left']),
                    "RightEnvy": len(envy['right']),
                    "LeftGini": gini[0],
                    "RightGini": gini[1],
                    "ExecutionTime": execution_time
                }
            )
            print(f"Alpha={alpha}: Match={result[-1]['Match']}, LeftEnvy={result[-1]['LeftEnvy']}, RightEnvy={result[-1]['RightEnvy']}, LeftGini={result[-1]['LeftGini']}, RightGini={result[-1]['RightGini']}, ExecutionTime={result[-1]['ExecutionTime']}")

    result_file_path = f'{result_dir}/experiment2_n{num_left}_m{num_right}_v{v_type}_lambda{lambda_value}_testcases{testcases}.json'

    with open(result_file_path, "w") as f:
        json.dump(result, f, indent=4)


def experiment3(
        result_dir: str = ".",
        num_left: int = 10,
        num_right: int = 10,
        v_type: str = 'log',
        lambda_value: float = 0.8,
        testcases: int = 10,
        solver: str = "CLARABEL",
        device: str = "cpu",
        sinkhorn_lambda: float = 200.0
) -> None:
    """
    Conducts experiments in the setting of "8.4 Synthetic Data Experiment III" in Tomita and Yokoyama (2025).

    Parameters
    ----------
    result_dir: str
        The directory to save the results.
    num_left: int
        The number of left users.
    num_right: int
        The number of right users.
    v_type: str
        The type of valuation functions ('log' or 'inv').
    lambda_value: float
        The lambda value for the preference generation.
    testcases: int
        The number of test cases to run.
    solver: str
        The CVXPY solver used in the IterLP, SW_LP and NSW_LP methods. "CLARABEL", "ECOS" or etc.
    device: str
        The device for the SW_Sinkhorn and NSW_Sinkhorn methods. "cpu", "cuda" or etc.
    sinkhorn_lambda: float
        Hyper parameter in the Sinkhorn algorithm used in the SW_Sinkhorn and NSW_Sinkhorn methods.
    """
    result = []
    for seed in range(testcases):
        print(f"\n========== Seed: {seed} ==========")
        m = Market(num_left=num_left, num_right=num_right, v_left_type=v_type, v_right_type=v_type)
        m.generate_preferences(pref_seed=seed, lambda_value=lambda_value)

        for method in ["Naive", "Prod", "IterLP", "TU", "SW_LP", "NSW_LP", "SW_Sinkhorn", "NSW_Sinkhorn"]:
            start_time = time.time()
            if method == "Naive":
                left_rec = naive(m.pref_left_to_right)
                right_rec = naive(m.pref_right_to_left)
            elif method == "Prod":
                left_rec = prod(m.pref_left_to_right, m.pref_right_to_left)
                right_rec = prod(m.pref_right_to_left, m.pref_left_to_right)
            elif method == "IterLP":
                left_rec, right_rec = iter_lp(m.pref_left_to_right, m.pref_right_to_left, solver=solver)
            elif method == "TU":
                left_rec, right_rec = tu_matching(m.pref_left_to_right, m.pref_right_to_left, output=False)
            elif method == "SW_LP":
                left_rec, right_rec = sw_maximize(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, solver=solver, output=False)
            elif method == "NSW_LP":
                left_rec, right_rec = nsw_maximize(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, solver=solver, output=False)
            elif method == "SW_Sinkhorn":
                left_rec, right_rec = sw_sinkhorn(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, device=device, sinkhorn_lambda=sinkhorn_lambda, output=False)
            elif method == "NSW_Sinkhorn":
                left_rec, right_rec = nsw_sinkhorn(m.pref_left_to_right, m.pref_right_to_left, m.v_left, m.v_right, device=device, sinkhorn_lambda=sinkhorn_lambda, output=False)
            else:
                raise ValueError("Invalid Method")
            execution_time = time.time()-start_time
            
            match_prob = m.get_match_prob(left_rec, right_rec)
            envy = m.check_envy(left_rec, right_rec, match_prob)
            gini = m.compute_gini(match_prob)
            result.append(
                {
                    "NumLeft": num_left,
                    "NumRight": num_right,
                    "VType": v_type,
                    "Lambda": lambda_value,
                    "Seed": seed,
                    "Method": method,
                    "Match": match_prob.sum(),
                    "LeftEnvy": len(envy['left']),
                    "RightEnvy": len(envy['right']),
                    "LeftGini": gini[0],
                    "RightGini": gini[1],
                    "ExecutionTime": execution_time
                }
            )
            print(f"{method}: Match={result[-1]['Match']}, LeftEnvy={result[-1]['LeftEnvy']}, RightEnvy={result[-1]['RightEnvy']}, LeftGini={result[-1]['LeftGini']}, RightGini={result[-1]['RightGini']}, ExecutionTime={result[-1]['ExecutionTime']}")

    result_file_path = f'{result_dir}/experiment3_n{num_left}_m{num_right}_v{v_type}_lambda{lambda_value}_testcases{testcases}.json'

    with open(result_file_path, "w") as f:
        json.dump(result, f, indent=4)
