"""Lessons 3.7-3.8 - ACT with the cameras taken out.

Structurally this is the real thing: a CVAE encoder that reads the demonstrated
action chunk, a transformer encoder over the observation tokens, and a decoder
with one learned query per chunk step. What is missing is the ResNet-18 image
backbone, because a laptop CPU cannot train that in a lesson-sized run. Every
tensor that would carry image tokens is marked, so you can see the hole.

Defaults here follow LeRobot's `configuration_act.py` at tag v0.6.1 (read
2026-08-09) wherever the shape allows, and are shrunk where they do not:

    LeRobot ACT              this file        why
    dim_model      512       256              CPU training budget
    n_heads        8         8                same
    dim_feedforward 3200     1024             CPU training budget
    n_encoder_layers 4       4                same
    n_decoder_layers 1       1                same (and see the lesson)
    n_vae_encoder_layers 4   4                same
    latent_dim     32        32               same
    kl_weight      10.0      10.0             same
    chunk_size     100       50               8-second episodes at 30 fps
    vision_backbone resnet18  none            no images here

Run:  python act_lite.py             shapes, parameter count, inference latency
      python act_lite.py --collapse what happens when the KL weight is zero
"""
import sys
import time

import torch
import torch.nn as nn


class ACTLite(nn.Module):
    def __init__(self, state_dim=6, action_dim=6, chunk_size=50, d_model=256,
                 n_heads=8, dim_feedforward=1024, n_encoder_layers=4,
                 n_decoder_layers=1, n_vae_encoder_layers=4, latent_dim=32,
                 dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # --- CVAE encoder: reads state AND the demonstrated chunk, emits z ----
        # It exists so the model has somewhere to put the variation between two
        # demonstrations of the same task that the observation cannot explain.
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.vae_state_proj = nn.Linear(state_dim, d_model)
        self.vae_action_proj = nn.Linear(action_dim, d_model)
        self.vae_pos = nn.Parameter(torch.zeros(1, chunk_size + 2, d_model))
        self.vae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                       batch_first=True),
            num_layers=n_vae_encoder_layers)
        self.to_latent = nn.Linear(d_model, 2 * latent_dim)   # mu and log-variance

        # --- observation encoder: what the policy sees at inference time ------
        # Two tokens here: the latent and the joint state. Real ACT adds one
        # token per spatial cell of every camera's ResNet-18 feature map.
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.obs_pos = nn.Parameter(torch.zeros(1, 2, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                       batch_first=True),
            num_layers=n_encoder_layers)

        # --- decoder: one learned query per step of the chunk -----------------
        self.queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                       batch_first=True),
            num_layers=n_decoder_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def encode_latent(self, state, action_chunk):
        """Training only. Returns (mu, logvar) of q(z | state, actions)."""
        B = state.shape[0]
        tokens = torch.cat([
            self.vae_cls.expand(B, -1, -1),
            self.vae_state_proj(state).unsqueeze(1),
            self.vae_action_proj(action_chunk),
        ], dim=1) + self.vae_pos
        out = self.vae_encoder(tokens)[:, 0]          # the [cls] slot
        mu, logvar = self.to_latent(out).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, state, z):
        B = state.shape[0]
        memory = torch.cat([
            self.latent_proj(z).unsqueeze(1),
            self.state_proj(state).unsqueeze(1),
            # image tokens would be concatenated here
        ], dim=1) + self.obs_pos
        memory = self.encoder(memory)
        out = self.decoder(self.queries.expand(B, -1, -1), memory)
        return self.action_head(out)                  # (B, chunk_size, action_dim)

    def forward(self, state, action_chunk=None):
        """Training passes the chunk; inference does not and z becomes zero."""
        if action_chunk is None:
            z = torch.zeros(state.shape[0], self.latent_dim, device=state.device)
            return self.decode(state, z), None, None
        mu, logvar = self.encode_latent(state, action_chunk)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(state, z), mu, logvar


def act_loss(pred, target, mu, logvar, kl_weight=10.0, mask=None):
    """L1 reconstruction plus the KL term that keeps z near the prior.

    `mask` is (B, chunk) and is False on padded steps at the end of an episode.
    Normalising by the number of REAL steps rather than by the padded tensor
    size is the fix LeRobot shipped in v0.6.0 (PR #3442); dividing by the padded
    size quietly scales the loss down for every sample near an episode boundary.
    """
    per_step = (pred - target).abs().mean(dim=-1)          # (B, chunk)
    if mask is None:
        recon = per_step.mean()
    else:
        recon = (per_step * mask).sum() / mask.sum().clamp(min=1)
    if mu is None:
        return recon, recon, torch.zeros((), device=pred.device)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()
    return recon + kl_weight * kl, recon, kl


def _report():
    torch.manual_seed(0)
    model = ACTLite()
    state = torch.randn(4, 6)
    chunk = torch.randn(4, 50, 6)

    pred, mu, logvar = model(state, chunk)
    print(f"training pass   state {tuple(state.shape)} + chunk {tuple(chunk.shape)}"
          f"  ->  {tuple(pred.shape)}, z is {tuple(mu.shape)}")
    total, recon, kl = act_loss(pred, chunk, mu, logvar)
    print(f"                loss {total.item():.3f} = L1 {recon.item():.3f} "
          f"+ 10.0 x KL {kl.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_a, mu_a, _ = model(state)
        pred_b, _, _ = model(state)
    print(f"inference pass  state {tuple(state.shape)} only        ->  "
          f"{tuple(pred_a.shape)}, z is a vector of {model.latent_dim} zeros "
          f"(no posterior: mu is {mu_a})")
    print(f"                two calls identical: {torch.equal(pred_a, pred_b)}  "
          f"(z is fixed at zero, so the policy is deterministic)")

    groups = {"CVAE encoder (training only)":
              ["vae_cls", "vae_state_proj", "vae_action_proj", "vae_pos",
               "vae_encoder", "to_latent"],
              "observation encoder": ["latent_proj", "state_proj", "obs_pos", "encoder"],
              "decoder + head": ["queries", "decoder", "action_head"]}
    print()
    for name, prefixes in groups.items():
        n = sum(p.numel() for k, p in model.named_parameters()
                if any(k.startswith(x) for x in prefixes))
        print(f"  {name:<32} {n:>10,} params")
    print(f"  {'total':<32} {sum(p.numel() for p in model.parameters()):>10,} params")

    with torch.no_grad():
        one = torch.randn(1, 6)
        for _ in range(5):
            model(one)
        best = min(_time_once(model, one) for _ in range(30))
    print(f"\n  one inference: {best*1000:.1f} ms  -> a 50-step chunk at 30 fps is "
          f"1.67 s of robot time, so the policy runs at "
          f"{best/1.667*100:.2f}% duty cycle")


def _time_once(model, one):
    t0 = time.perf_counter()
    model(one)
    return time.perf_counter() - t0


def _collapse(steps=400, kl_weights=(0.0, 10.0)):
    """Show the latent turning into a cheat channel when nothing penalises it.

    64 fixed random states, each paired with its own fixed random chunk. There
    are two routes to a low training loss: memorise state -> chunk, which
    survives deployment, or write the chunk into z and read it back, which does
    not. Both look identical on the training curve. Column 4 runs the deployed
    path, where z is zero, and separates them.
    """
    torch.manual_seed(0)
    state = torch.randn(64, 6)
    chunk = torch.randn(64, 12, 6)
    print(f"{'kl_weight':>10} {'train L1':>10} {'KL':>10} {'inference L1':>14}")
    for weight in kl_weights:
        torch.manual_seed(0)
        model = ACTLite(chunk_size=12, d_model=64, dim_feedforward=128,
                        n_encoder_layers=2, n_vae_encoder_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(steps):
            pred, mu, logvar = model(state, chunk)
            loss, recon, kl = act_loss(pred, chunk, mu, logvar, kl_weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            deployed, _, _ = model(state)
            gap = (deployed - chunk).abs().mean().item()
        print(f"{weight:>10.1f} {recon.item():>10.4f} {kl.item():>10.2f} {gap:>14.4f}")
    print("  with no KL penalty the training column looks better and the "
          "deployed column is worse")


if __name__ == "__main__":
    if "--collapse" in sys.argv:
        _collapse()
    else:
        _report()
