"""CoT-assisted Q-SFT Data Processor with V-CoT integration."""
import torch
import numpy as np
from typing import Dict, Any, List
from swift.llm import MessagesPreprocessor
from swift.utils import get_logger

logger = get_logger()

class QSFTRewardCalculator:
    """Compute rewards and Q-values with CoT-assisted estimation."""
    def __init__(self, step_penalty=-0.01, progress_scale=0.5, 
                 success_reward=1.0, failure_penalty=-0.2):
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
    def compute_rewards(self, trajectory: List[dict], success: bool) -> List[float]:
        T = len(trajectory)
        rewards = []
        for t, step in enumerate(trajectory):
            r = self.step_penalty
            if t > 0 and 'distance_to_goal' in step and 'distance_to_goal' in trajectory[t-1]:
                d_prev = trajectory[t-1]['distance_to_goal']
                d_curr = step['distance_to_goal']
                r += (d_prev - d_curr) * self.progress_scale
            if t == T - 1 and step.get('action') == 'stop':
                r += self.success_reward if success else self.failure_penalty
            rewards.append(r)
        return rewards
    
    def compute_q_values(self, rewards: List[float], gamma: float = 0.95) -> List[float]:
        q_values = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            q_values.insert(0, G)
        return q_values


class UMMCOTPreprocessorQSFT(MessagesPreprocessor):
    """Extended preprocessor with Q-SFT and V-CoT support."""
    def __init__(self, q_gamma=0.95, **kwargs):
        super().__init__(**kwargs)
        self.q_gamma = q_gamma
        self.reward_calc = QSFTRewardCalculator()
    
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        processed = super().preprocess(row)
        trajectory = row.get('trajectory')
        success = row.get('success', False)
        
        if trajectory is not None:
            q_weights = self._compute_q_weights(processed, trajectory, success)
            var_mask = self._extract_var_mask(processed)
            for branch_name in processed.keys():
                if isinstance(processed[branch_name], dict):
                    processed[branch_name]['q_weights'] = q_weights
                    if var_mask is not None:
                        processed[branch_name]['var_mask'] = var_mask
        return processed
    
    def _compute_q_weights(self, processed, trajectory, success):
        rewards = self.reward_calc.compute_rewards(trajectory, success)
        q_values = self.reward_calc.compute_q_values(rewards, self.q_gamma)
        non_cot = processed.get('Non_CoT', list(processed.values())[0] if processed else {})
        labels = non_cot.get('labels', [])
        seq_len = len(labels)
        q_weights = torch.zeros(seq_len)
        action_positions = self._find_action_positions(labels)
        for i, pos in enumerate(action_positions):
            if i < len(q_values) and pos < seq_len:
                q_weights[pos] = q_values[i]
        return q_weights
    
    def _find_action_positions(self, labels):
        positions = []
        for i, label in enumerate(labels):
            if label >= 0:
                positions.append(i)
        return positions
    
    def _extract_var_mask(self, processed):
        """Extract mask for <var>...</var> token positions."""
        vcot = processed.get('V_CoT')
        if vcot is None:
            return None
        # This would need tokenizer access to find exact positions
        # For now, placeholder - actual implementation needs token IDs
        return None
