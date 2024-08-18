# Fair Reciprocal Recommendation in Matching Markets

This repository contains the algorithms and synthetic data generation codes used in the experimental section of the paper 'Fair Reciprocal Recommendation in Matching Markets' by Yoji Tomita and Tomohiko Yokoyama, RecSys2024.

## Requirements

If you can use [Rye](https://github.com/astral-sh/rye), you can set up the environment just by running the following command:
```
rye sync
```

If you need to set up the Python environment manually, we require the followings:
- Python >= 3.8
- numpy >= 2.0.1
- cvxpy >= 1.5.2
- jupyter >= 1.0.0


## Files

In the `src` directory, there are the following files:

 - `market.py`: The class definition of the matching markets, including the synthetic preference data generation and the checking function of the envy-freeness.
 - `utils.py`: The utility functions.
 - `naive.py`: The algorithm for the naive method.
 - `prod.py`: The algorithm for the prod method.
 - `tu_matching.py`: The algorithm for the TU-matching method.
 - `iter_lp.py`: The algorithm for the iterative LP method.
 - `alternate_fw.py`: The proposed algorithm for the NSW and SW maximization via the alternate Frank-Wolfe method.

## Example Usage
```python
from src.market import Market
from src.alternate_fw import nsw_maximize

# Make a market with 30 left agents and 20 right agents, and generate preferences
mkt = Market(num_left=30, num_right=20)
mkt.generate_preferences(pref_seed=0)

# Run the NSW maximization algorithm
policy_for_left, policy_for_right = nsw_maximize(m.pref_left_to_right, m.pref_right_to_left, v_left=m.v_left, v_right=m.v_right)

# Compute the matching probabilities and check the envy-freeness
match_prob = mkt.get_match_prob(policy_for_left, policy_for_right)
print("Expected number of matches:", match_prob.sum())
envy = mkt.check_envy(policy_for_left, policy_for_right, match_prob=match_prob)
print("Number of envies for left agents:", len(envy["left"]))
print("Number of envies for right agents:", len(envy["right"]))
```
See `notebooks/example.ipynb` for more detailed examples.

## License
This repository is licensed under the MIT License.
