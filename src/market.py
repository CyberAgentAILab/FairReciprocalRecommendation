import numpy as np
from utils import generate_examination_vector

class Market:
    """Market class which holds preferences and examination vectors of two sides of agents.

    Attributes
    ----------
    num_left : int
        Number of left side agents.
    num_right : int
        Number of right side agents.
    pref_left_to_right : np.ndarray
        Preferences of left side agents to right side agents.
    pref_right_to_left : np.ndarray
        Preferences of right side agents to left side agents.
    v_left : np.ndarray
        Examination vector of left side agents.
    v_right : np.ndarray
        Examination vector of right side agents.
    """
    def __init__(
            self,
            num_left: int,
            num_right: int,
            pref_left_to_right: np.ndarray | None = None,
            pref_right_to_left: np.ndarray | None = None,
            v_left_type: str = "inv",
            v_right_type: str = "inv"
    ) -> None:
        """Initialize Market class.

        Parameters
        ----------
        num_left : int
            Number of left side agents.
        num_right : int
            Number of right side agents.
        pref_left_to_right : np.ndarray, optional
            Preferences of left side agents to right side agents. Shape must be (num_left, num_right).
        pref_right_to_left : np.ndarray, optional
            Preferences of right side agents to left side agents. Shape must be (num_right, num_left).
        v_left_type : str, optional
            Type of examination vector for left side agents. It must be one of "inv", "log", "exp", "k_1", "uni", or "lin". Default is "inv".
        v_right_type : str, optional
            Type of examination vector for right side agents. It must be one of "inv", "log", "exp", "k_1", "uni", or "lin". Default is "inv".
        """
        self.num_left: int = num_left
        self.num_right: int = num_right
        self.pref_left_to_right: np.ndarray
        self.pref_right_to_left: np.ndarray
        self.v_left = generate_examination_vector(v_type=v_left_type, num_rank=self.num_right)
        self.v_right = generate_examination_vector(v_type=v_right_type, num_rank=self.num_left)

        if pref_left_to_right is None:
            self.pref_left_to_right = np.zeros(shape=(self.num_left, self.num_right))    # Initialize with zeros if not given
        else:
            assert pref_left_to_right.shape == (self.num_left, self.num_right)
            self.pref_left_to_right = pref_left_to_right.copy()

        if pref_right_to_left is None:
            self.pref_right_to_left = np.zeros(shape=(self.num_right, self.num_left))    # Initialize with zeros if not given
        else:
            assert pref_right_to_left.shape == (self.num_right, self.num_left)
            self.pref_right_to_left = pref_right_to_left.copy()


    def generate_preferences(
            self,
            pref_seed: int | None = None,
            lambda_value: float = 0.5
    ) -> None:
        """Generate preferences randomly and update the attributes.

        Parameters
        ----------
        pref_seed : int, optional
            Seed for random number generator. Default is None.
        lambda_value : float, optional
            Weight for congestion value. Default is 0.5.
        """
        np.random.seed(seed=pref_seed)
        self.pref_left_to_right = np.random.random((self.num_left, self.num_right))    # Gerenate random preferences
        self.pref_right_to_left = np.random.random((self.num_right, self.num_left)) 

        right_general_popularity = np.tile(np.linspace(1, 0, self.num_right), reps=(self.num_left, 1))    # General popularity of right side agents
        self.pref_left_to_right = np.clip((1 - lambda_value) * self.pref_left_to_right + lambda_value * right_general_popularity, 0.0, 1.0)    # Take convex combination

        left_general_popularity = np.tile(np.linspace(1, 0, self.num_left), reps=(self.num_right, 1))    # General popularity of left side agents
        self.pref_right_to_left = np.clip((1 - lambda_value) * self.pref_right_to_left + lambda_value * left_general_popularity, 0.0, 1.0)    # Take convex combination


    def get_match_prob(
            self,
            stochastic_policy_for_left: np.ndarray,
            stochastic_policy_for_right: np.ndarray,
    ) -> np.ndarray:
        """Return the probability of matching between left and right side agents given stochastic policies.

        Parameters
        ----------
        stochastic_policy_for_left : np.ndarray
            Stochastic policy for left side agents. Shape must be (num_left, num_right, num_right).
        stochastic_policy_for_right : np.ndarray
            Stochastic policy for right side agents. Shape must be (num_right, num_left, num_left).
        
        Returns
        -------
        np.ndarray
            Match probabilities between left and right side agents. Shape is (num_left, num_right).
        """
        assert stochastic_policy_for_left.shape == (self.num_left, self.num_right, self.num_right)
        assert stochastic_policy_for_right.shape == (self.num_right, self.num_left, self.num_left)

        left_like_prob = self.get_like_prob(self.pref_left_to_right, self.v_left, stochastic_policy_for_left)    # Probabilities of left side agents liking right side agents (num_left, num_right)
        right_likeprob = self.get_like_prob(self.pref_right_to_left, self.v_right, stochastic_policy_for_right)    # Probabilities of right side agents liking left side agents (num_right, num_left)

        match_prob = left_like_prob * right_likeprob.T    # Match probabilities between left and right side agents (num_left, num_right)
        return match_prob


    def get_like_prob(
            self,
            preferences: np.ndarray,
            v_vector: np.ndarray,
            stochastic_policy: np.ndarray
    ) -> np.ndarray:
        """Return the probability of liking given preferences, examination vector, and stochastic policy.

        Parameters
        ----------
            preferences : np.ndarray
                Preferences of agents.
            v_vector : np.ndarray
                Examination vector of agents.
            stochastic_policy : np.ndarray
                Stochastic policy of agents.
        
        Returns
        -------
        np.ndarray
            Probabilities of liking. Shape is (num_agents, num_others).
        """
        examination_prob = stochastic_policy @ v_vector
        return np.clip(preferences * examination_prob, 0.0, 1.0)


    def expected_matches_of_one_agent(
            self,
            agent_type: str,
            preferences: np.ndarray,
            stochastic_policy: np.ndarray,
            opposite_preference: np.ndarray,
            opposite_stochastic_policy: np.ndarray
    ) -> np.float64:
        """Return the expected number of matches of one agent given preferences, examination vector, and stochastic policy.
        This function is used in check_envy.

        Parameters
        ----------
        agent_type : str
            Type of agent. It must be "left" or "right".
        preferences : np.ndarray
            Preferences of one agent. Shape must be (num_others,).
        stochastic_policy : np.ndarray
            Stochastic policy of one agent. Shape must be (num_others, num_others).
        opposite_preference : np.ndarray
            Preferences of opposite agents to one agent. Shape must be (num_others,).
        opposite_stochastic_policy : np.ndarray
            Stochastic policy of opposite agents. Shape must be (num_others, num_agent).
            i,j-th element is the probability with which the agent is recommended to i-th opposite agent in j-th position.
        
        Returns
        -------
        np.float64
            Expected number of matches of one agent
        """
        if agent_type == "left":
            v_self, v_opposite = self.v_left, self.v_right
        elif agent_type == "right":
            v_self, v_opposite = self.v_right, self.v_left
        else:
            raise ValueError('"agent_type" must be "left" or "right".')

        like_prob = self.get_like_prob(preferences, v_self, stochastic_policy)    # Probabilities of the agent liking opposite agents (num_others)
        opposite_like_prob = self.get_like_prob(opposite_preference, v_opposite, opposite_stochastic_policy)    # Probabilities of opposite agents liking the agent (num_others)
        return like_prob @ opposite_like_prob


    def check_envy(
            self,
            stochastic_policy_for_left: np.ndarray,
            stochastic_policy_for_right: np.ndarray,
            match_prob: np.ndarray | None = None,
            tol: np.float64 = 1e-9
    ) -> dict[str, list[tuple]]:
        """Check envy of agents given stochastic policies.

        Parameters
        ----------
        stochastic_policy_for_left : np.ndarray
            Stochastic policy for left side agents. Shape must be (num_left, num_right, num_right).
        stochastic_policy_for_right : np.ndarray
            Stochastic policy for right side agents. Shape must be (num_right, num_left, num_left).
        match_prob : np.ndarray, optional
            Match probabilities between left and right side agents. Shape is (num_left, num_right). Default is None.
        tol : np.float64, optional
            Tolerance for comparison. Default is 1e-9.
        
        Returns
        -------
        envy : dict[str, list[tuple]]
            Envy of agents. Keys are "left" and "right".
            If (i, j) in envy["left"], i-th left agent envies j-th left agent.
        """
        if match_prob is None:
            match_prob = self.get_match_prob(stochastic_policy_for_left, stochastic_policy_for_right)    # Compute match probabilities if not given

        envy: dict[str, list[tuple]] = {"left": [], "right": []}    # Initialize envy dictionary

        # Check envy of left side agents
        for i in range(self.num_left):
            match_prob_i = np.sum(match_prob[i])    # Expected number of matches of i-th left agent
            for j in range(self.num_left):
                if i == j:
                    continue
                tmp_match_prob = self.expected_matches_of_one_agent(    # Expected number of matches of i-th agent if i is recommended to right agents in j-th position
                    agent_type="left",
                    preferences=self.pref_left_to_right[i],
                    stochastic_policy=stochastic_policy_for_left[i],
                    opposite_preference=self.pref_right_to_left[:, i],
                    opposite_stochastic_policy=stochastic_policy_for_right[:, j, :]
                )
                if match_prob_i < tmp_match_prob - tol:
                    envy["left"].append((i, j))    # i-th left agent envies j-th left agent

        # Check envy of right side agents
        for i in range(self.num_right):
            match_prob_i = np.sum(match_prob[:, i])    # Expected number of matches of i-th right agent
            for j in range(self.num_right):
                if i == j:
                    continue
                tmp_match_prob = self.expected_matches_of_one_agent(    # Expected number of matches of i-th agent if i is recommended to left agents in j-th position
                    agent_type="right",
                    preferences=self.pref_right_to_left[i],
                    stochastic_policy=stochastic_policy_for_right[i],
                    opposite_preference=self.pref_left_to_right[:, i],
                    opposite_stochastic_policy=stochastic_policy_for_left[:, j, :]
                )
                if match_prob_i < tmp_match_prob - tol:
                    envy["right"].append((i, j))    # i-th right agent envies j-th right agent

        return envy
