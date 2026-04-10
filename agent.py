import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def tanh01(x):
    return (torch.tanh(x) + 1.0) * 0.5

def init_decoder_constant(model, constant=1.0):
    for name, param in model.decoder.named_parameters():
        if "weight" in name:
            nn.init.constant_(param, 0.0)
        elif "bias" in name:
            nn.init.constant_(param, constant)

class ImgOnlyParamAgent(nn.Module):
    def __init__(self, output_dim, img_channels=3, img_base=32, latent_dim=128):
        super().__init__()
        ch = img_base
        self.down_sample = nn.AdaptiveAvgPool2d((64, 64))
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, ch, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(ch * 8, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, output_dim),
        )

    def forward(self, img):
        img = self.down_sample(img)
        feat = self.enc3(self.enc2(self.enc1(img)))
        gap = F.adaptive_avg_pool2d(feat, 1).view(img.size(0), -1)
        gmp = F.adaptive_max_pool2d(feat, 1).view(img.size(0), -1)
        hidden = F.relu(self.fc1(torch.cat([gap, gmp], dim=1)), inplace=True)
        return tanh01(self.decoder(self.fc2(hidden)))

class ActionAgentGRU(nn.Module):
    def __init__(self, module_num, output_dim):
        super().__init__()
        self.hidden_dim = 64
        self.module_num = module_num
        self.output_dim = output_dim
        self.sos_token_id = module_num + 1
        self.eos_token_id = module_num
        self.max_len = module_num
        self.step_dim = 16
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.Linear(self.hidden_dim * 2, output_dim)
        self.token_embed = nn.Embedding(module_num + 2, self.hidden_dim)
        self.step_embed = nn.Embedding(self.max_len + 1, self.step_dim)
        self.step2film = nn.Sequential(
            nn.Linear(self.step_dim, self.hidden_dim * 4),
            nn.Tanh(),
        )

    @staticmethod
    def _mask_already_selected(logits, selected):
        masked = logits.clone()
        for action in selected:
            if 0 <= action < masked.size(-1):
                masked[0, action] = float("-inf")
        return masked

    @staticmethod
    def _exp_decay_tau(step, tau_max, tau_min, half_life_steps):
        decay = math.log(2.0) / max(1, int(half_life_steps))
        return max(1e-6, float(tau_min + (tau_max - tau_min) * math.exp(-decay * max(0, int(step)))))

    def _decode_single(self, temperature, device, is_val=False):
        hidden = torch.zeros(2, 1, self.hidden_dim, device=device)
        inputs = torch.full((1, 1), self.sos_token_id, dtype=torch.long, device=device)
        actions = []
        log_probs = []
        step_table = self.step_embed(torch.arange(0, self.max_len + 1, device=device))
        for step_index in range(self.max_len):
            embedded = self.token_embed(inputs)
            gru_out, hidden = self.gru(embedded, hidden)
            hidden_state = gru_out[:, -1, :]
            gamma, beta = self.step2film(step_table[min(step_index, self.max_len)].unsqueeze(0)).chunk(2, dim=-1)
            hidden_state = hidden_state * (1 + gamma) + beta
            logits = self.decoder(hidden_state)
            logits = self._mask_already_selected(logits, actions) / temperature
            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                probs = F.softmax(self.decoder(hidden_state), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = torch.argmax(probs, dim=-1) if is_val else dist.sample()
            log_probs.append(dist.log_prob(action))
            action_id = action.item()
            actions.append(action_id)
            if action_id == self.eos_token_id:
                break
            inputs = action.unsqueeze(0)
        return actions, log_probs

    def forward(self, step, batch_size, is_val=False, tau_max=2.5, tau_min=0.2, half_life_steps=3000):
        device = self.decoder.weight.device
        temperature = self._exp_decay_tau(step, tau_max, tau_min, half_life_steps)
        actions_batch = []
        log_probs_batch = []
        for _ in range(batch_size):
            actions, log_probs = self._decode_single(temperature=temperature, device=device, is_val=is_val)
            actions_batch.append(actions)
            log_probs_batch.append(log_probs)
        return actions_batch, log_probs_batch

class Agent(nn.Module):
    def __init__(self, output_path, filters):
        super().__init__()
        self.output_path = output_path
        self.filter_arr = filters
        self.filter = nn.ModuleList(filters)
        self.stage_num = len(self.filter_arr)
        self.param_num_arr = [module.param_num for module in self.filter]
        self.param_num = sum(self.param_num_arr)
        self.module_cnt = len(self.filter)
        self.eos_token = self.module_cnt
        self.max_bri = 0.9
        self.action_agent = ActionAgentGRU(module_num=self.module_cnt, output_dim=self.module_cnt + 1)
        self.param_net = ImgOnlyParamAgent(output_dim=self.param_num, img_channels=3)
        init_decoder_constant(self.action_agent, constant=1.0)

    def apply_filter(self, image, selected_order, params):
        output = image.unsqueeze(0)
        penalty = output.new_tensor(0.0)
        for action in selected_order:
            if action == self.eos_token:
                break
            start = sum(self.param_num_arr[:action])
            end = start + self.param_num_arr[action]
            output = self.filter[action](output, params[start:end])
        retouch_mean = torch.mean(output, dim=(1, 2, 3)).unsqueeze(-1)
        if torch.any((retouch_mean < 0.01) | (retouch_mean > self.max_bri)):
            penalty = penalty + (
                (
                    torch.clamp(0.01 - retouch_mean, min=0) * 100
                    + torch.clamp(retouch_mean - self.max_bri, min=0)
                )
                * 10
            ).mean()
        return output, penalty

    def forward(self, x, num_steps, writer=None, is_val=False, save_path=""):
        batch_size = x.shape[0]
        selected_orders, selected_prob = self.action_agent(step=num_steps, batch_size=batch_size, is_val=is_val)
        selected_prob = torch.stack(
            [
                torch.stack(prob).sum() if len(prob) else x.new_tensor(0.0)
                for prob in selected_prob
            ]
        )
        params = self.param_net(x)
        penalty = x.new_zeros(batch_size)
        output = []
        for index, (image, order) in enumerate(zip(x, selected_orders)):
            filtered, stop_penalty = self.apply_filter(image, order, params[index])
            penalty[index] += stop_penalty
            output.append(filtered)
        early_stop_penalty = x.new_zeros(batch_size)
        return {
            "output": torch.cat(output, dim=0),
            "selected_prob": selected_prob,
            "penalty": penalty,
            "early_stop_penalty": early_stop_penalty,
        }

    @torch.no_grad()
    def inference(self, x, save_path="", x_origin=None):
        source = x_origin if x_origin is not None else x
        source = source.to(next(self.parameters()).device)
        selected_orders, _ = self.action_agent(step=20_000_000, batch_size=source.shape[0], is_val=True)
        params = self.param_net(source)
        outputs = []
        images_all = []
        params_all = []
        names_all = []
        for index, (image, order) in enumerate(zip(source, selected_orders)):
            current = image.unsqueeze(0)
            steps = [current]
            selected_params = []
            selected_names = []
            for action in order:
                if action == self.eos_token:
                    break
                start = sum(self.param_num_arr[:action])
                end = start + self.param_num_arr[action]
                module = self.filter[action]
                module_params = params[index][start:end]
                selected_params.append(module.get_param(module_params) if hasattr(module, "get_param") else module_params)
                selected_names.append(module.name)
                current = module(current, module_params)
                steps.append(current)
            outputs.append(current)
            images_all.append(steps)
            params_all.append(selected_params)
            names_all.append(selected_names)
        return {
            "output": torch.cat(outputs, dim=0),
            "imgs": images_all,
            "params_return": params_all,
            "names_return": names_all,
        }
