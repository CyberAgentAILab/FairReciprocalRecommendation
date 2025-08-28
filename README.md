# Fair Reciprocal Recommendation in Matching Markets

This repository contains the algorithms and synthetic data generation codes used in the experimental sections of the following papers:
- [1] Yoji Tomita and Tomohiko Yokoyama. 2024. "Fair Reciprocal Recommendation in Matching Markets." Proceedings of the 18th ACM Conference on Recommender Systems (RecSys'24).
- [2] Yoji Tomita and Tomohiko Yokoyama. 2025. "Balancing Fairness and High Match Rates in Reciprocal Recommender Systems: A Nash Social Welfare Approach." Working Paper.

## Requirements

If you can use [Rye](https://github.com/astral-sh/rye), you can set up the environment just by running the following command:
```
rye sync
```

If you need to set up the Python environment manually, we require the followings:
- Python >= 3.9
- numpy >= 2.0.1
- cvxpy >= 1.5.2
- jupyter >= 1.0.0
- torch >= 2.8.0


## Files

In the [`src`](https://github.com/CyberAgentAILab/FairReciprocalRecommendation/tree/main/src) directory, there are the following files:

 - `market.py`: The class definition of the matching markets, including the synthetic preference data generation and the checking function of the envy-freeness.
 - `utils.py`: The utility functions.
 - `naive.py`: The algorithm for the naive method (baseline).
 - `prod.py`: The algorithm for the prod method (baseline).
 - `tu_matching.py`: The algorithm for the TU-matching method (baseline).
 - `iter_lp.py`: The algorithm for the iterative LP method (baseline).
 - `alternate_fw.py`
   - `sw_maximize`: The alogirhtm for the SW maximization via the alternate Frank-Wolfe method (proposed in Section 3 of Tomita and Yokoyama [1]).
   - `nsw_maximize`: The alogirhtm for the NSW maximization via the alternate Frank-Wolfe method (proposed in Section 4 of Tomita and Yokoyama [1]).
   - `alpha_sw_maximize`: The algorithm for the α-SW maximization method (proposed in Section 6 of Tomita and Yokoyama [2]).
 - `alternate_fw_sinkhorn`
   - `sw_sinkhorn` : The approximate algorithm for the SW maximization via the Sinkhorn algorithm (proposed in Section 7 of Tomita and Yokoyama [2]).
   - `nsw_sinkhorn`: The approximate algorithm for the NSW maximization via the Sinkhorn algorithm (proposed in Section 7 of Tomita and Yokoyama [2]).
 - `experiments.py`: Conduct synthetic data experiments (Section 5.1 of Tomita and Yokoyama [1], Section 8.2-8.4 of Tomita and Yokoyama [2]).

## Example Usage
```
cd src
```
```python
from market import Market
from alternate_fw import nsw_maximize

# Make a market with 30 left agents and 20 right agents, and generate preferences
mkt = Market(num_left=30, num_right=20)
mkt.generate_preferences(pref_seed=0)

# Run the NSW maximization algorithm
policy_for_left, policy_for_right = nsw_maximize(mkt.pref_left_to_right, mkt.pref_right_to_left, mkt.v_left, mkt.v_right)

# Compute the matching probabilities and check the envy-freeness
match_prob = mkt.get_match_prob(policy_for_left, policy_for_right)
print("Expected number of matches:", match_prob.sum())
envy = mkt.check_envy(policy_for_left, policy_for_right, match_prob=match_prob)
print("Number of envies for left agents:", len(envy["left"]))
print("Number of envies for right agents:", len(envy["right"]))
```
For more detailed examples, see:
 - [`notebooks/example.ipynb`](https://github.com/CyberAgentAILab/FairReciprocalRecommendation/blob/main/notebooks/example.ipynb) for basic usages of the proposed algorithms.
 - [`notebooks/experiments.ipynb`](https://github.com/CyberAgentAILab/FairReciprocalRecommendation/blob/main/notebooks/experiments.ipynb) for the synthetic data experiments.

## License
This repository is licensed under the MIT License.
