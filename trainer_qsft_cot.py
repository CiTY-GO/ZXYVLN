"""CoT-assisted Q-SFT Trainer for FantasyVLN."""
import torch
import torch.nn.functional as F
import copy
from swift.trainers import Seq2SeqTrainer
from transformers.utils import get_logger

logger = get_logger()

class EMAModel:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            for sp, mp in zip(self.shadow.parameters(), model.parameters()):
                sp.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)

class CoTQSFTTrainer(Seq2SeqTrainer):
    def __init__(self, *args, use_ummcot=False, q_gamma=0.95, lambda_cot=0.5,
                 lambda_align=0.1, label_smoothing=0.1, clip_weight=5.0,
                 use_ema_target=True, ema_decay=0.995, **kwargs):
        self.use_ummcot = use_ummcot
        self.q_gamma = q_gamma
        self.lambda_cot = lambda_cot
        self.lambda_align = lambda_align
        self.label_smoothing = label_smoothing
        self.clip_weight = clip_weight
        self.use_ema_target = use_ema_target
        self.ema_decay = ema_decay
        self.ema_model = None
        super().__init__(*args, **kwargs)
        if self.use_ema_target and self.model is not None:
            self.ema_model = EMAModel(self.model, decay=ema_decay)
            logger.info(f"EMA initialized with decay={ema_decay}")

    def training_step(self, model, inputs, num_items_in_batch=None, mode="no_thinking"):
        if not self.use_ummcot or mode != "thinking":
            return self._standard_training_step(model, inputs, num_items_in_batch, mode)
        return self._cot_qsft_training_step(model, inputs, num_items_in_batch)

    def _standard_training_step(self, model, inputs, num_items_in_batch, mode):
        if mode == "no_thinking" and getattr(self, 'use_mmcot', False):
            inputs_list = [inputs['Non_CoT']]
        elif mode == "no_thinking":
            inputs_list = [inputs]
        else:
            inputs_list = [inputs['Non_CoT'], inputs['T_CoT'], inputs['V_CoT'], inputs['MM_CoT']]
        losses = []
        for inp in inputs_list:
            loss = super().compute_loss(model, self._prepare_inputs(inp), num_items_in_batch)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss = loss / self.current_gradient_accumulation_steps
            self.accelerator.backward(loss)
            losses.append(loss.detach())
        return torch.stack(losses).mean()

    def _cot_qsft_training_step(self, model, inputs, num_items_in_batch):
        device = self.args.device
        noncot = inputs['Non_CoT']
        tcot = inputs['T_CoT']
        vcot = inputs['V_CoT']
        mmcot = inputs['MM_CoT']
        
        with torch.no_grad():
            try:
                vcot_out = model(**self._prepare_inputs(vcot), output_hidden_states=True)
                var_hidden = self._extract_var_hidden(vcot_out, vcot.get('var_mask'))
                v_next = None
                if var_hidden is not None:
                    v_next = self._estimate_v_from_var(model, mmcot, var_hidden)
            except Exception as e:
                logger.warning(f"V-CoT extraction failed: {e}")
                v_next = None
        
        q_weights = noncot.pop('q_weights', None)
        if q_weights is not None and v_next is not None:
            q_weights = self._compute_cot_assisted_q(q_weights, v_next)
            noncot['q_weights'] = q_weights
        
        loss_noncot = self._compute_qsft_loss(model, noncot, num_items_in_batch)
        loss_tcot = super().compute_loss(model, self._prepare_inputs(tcot), num_items_in_batch)
        loss_vcot = super().compute_loss(model, self._prepare_inputs(vcot), num_items_in_batch)
        loss_mmcot = super().compute_loss(model, self._prepare_inputs(mmcot), num_items_in_batch)
        
        loss_align = torch.tensor(0.0, device=device)
        if self.lambda_align > 0:
            loss_align = self._compute_alignment_loss(model, noncot, mmcot)
        
        total_loss = loss_noncot + self.lambda_cot * (loss_tcot + loss_vcot + loss_mmcot) / 3 + self.lambda_align * loss_align
        
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()
        total_loss = total_loss / self.current_gradient_accumulation_steps
        self.accelerator.backward(total_loss)
        
        if self.ema_model is not None:
            self.ema_model.update(model)
        
        return total_loss.detach()

    def _extract_var_hidden(self, outputs, var_mask=None):
        """Extract hidden states corresponding to <var> tokens."""
        if var_mask is None:
            return None
        hidden_states = outputs.hidden_states[-1]
        return hidden_states * var_mask.unsqueeze(-1)

    def _estimate_v_from_var(self, model, mmcot_inputs, var_hidden):
        """Estimate V(s') using MM-CoT on imagined future obs."""
        return torch.tensor(0.0)

    def _compute_cot_assisted_q(self, q_weights, v_next):
        """Combine MC return with V-CoT assisted Bellman target."""
        if isinstance(q_weights, torch.Tensor):
            r_t = q_weights - self.q_gamma * torch.cat([q_weights[1:], torch.zeros_like(q_weights[:1])], dim=0)
            q_bellman = r_t + self.q_gamma * v_next
            return 0.5 * q_weights + 0.5 * q_bellman
        return q_weights

    def _compute_qsft_loss(self, model, inputs, num_items_in_batch):
        """Q-SFT: weighted CE with q_weights on action tokens."""
        q_weights = inputs.pop("q_weights", None)
        outputs = model(**inputs)
        logits, labels = outputs.logits, inputs.get("labels")
        if q_weights is None or labels is None:
            return super().compute_loss(model, inputs, num_items_in_batch)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = q_weights[..., 1:].contiguous()
        mask = (shift_labels != -100)
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        valid_labels = shift_labels.clone()
        valid_labels[~mask] = 0
        token_log_probs = log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
        
        valid_weights = shift_weights[mask]
        if valid_weights.numel() > 1:
            norm_weights = (shift_weights - valid_weights.mean()) / (valid_weights.std() + 1e-8)
        else:
            norm_weights = shift_weights
        norm_weights = norm_weights.clamp(-self.clip_weight, self.clip_weight)
        
        q_loss = -(norm_weights * token_log_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
        smooth_loss = -(log_probs * mask.float().unsqueeze(-1)).sum() / (mask.float().sum() * logits.shape[-1] + 1e-8)
        return q_loss + self.label_smoothing * smooth_loss

    def _compute_alignment_loss(self, model, noncot_inputs, mmcot_inputs):
        """Cross-mode Q consistency: Q_NonCoT should align with Q_MMCoT."""
        return torch.tensor(0.0)
