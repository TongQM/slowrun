"""Canonical cell -> log-glob map for the ALIGNED dynamics grid (2026-09-09 launch).

One place to say which run represents each (depth, width) cell, so every figure
script fits the same data. All cells: P=100M, lambda=0, constant LR, 40 epochs,
val every 152 steps, model 0 (seed 42), full architecture with the depth-scaled
residual paths (unlimited/train.py at commit 9226bde or later).

  al_dyn_*   single-model re-runs and fills from launch_aligned_grid.sh
  al_ens_*_5 model 0 of an ensemble cell: init_shuffle array task 5 = model 0,
             whose seed (42) and data seed (42) are identical to the init model 0,
             so it is the same run as an al_dyn_ cell would have been
  fd_*       the L=12 row, trained before the residual-path correction, which is
             exactly the identity at L=L_base=12 -- unaffected, kept as is

18 cells: the 4x4 factorial L in {6,12,18,24} x W in {384,768,1152,1536}, plus
the depth column extension d48/768 and d60/768.
"""

ALIGNED_CELLS = {
    (6, 384): "al_dyn_d6_w384_*_0.out",       (6, 768): "al_dyn_d6_w768_*_0.out",
    (6, 1152): "al_dyn_d6_w1152_*_0.out",     (6, 1536): "al_ens_d6_w1536_*_5.out",
    (12, 384): "fd_train_d12_w384_*_0.out",   (12, 768): "fd_train_d12_w768_*_0.out",
    (12, 1152): "fd_widthext_d12_w1152_*_0.out", (12, 1536): "fd_widthext_d12_w1536_*_0.out",
    (18, 384): "al_dyn_d18_w384_*_0.out",     (18, 768): "al_dyn_d18_w768_*_0.out",
    (18, 1152): "al_dyn_d18_w1152_*_0.out",   (18, 1536): "al_dyn_d18_w1536_*_0.out",
    (24, 384): "al_dyn_d24_w384_*_0.out",     (24, 768): "al_ens_d24_w768_*_5.out",
    (24, 1152): "al_dyn_d24_w1152_*_0.out",   (24, 1536): "al_dyn_d24_w1536_*_0.out",
    (48, 768): "al_dyn_d48_w768_*_0.out",     (60, 768): "al_dyn_d60_w768_*_0.out",
}

# The same recipe at P=20M (df=0.2, 50 epochs, val every 152 steps = one epoch there,
# the same 19.9M-token grid as the 100M runs). 152.5 optimizer steps per epoch.
P20_CELLS = {
    (6, 768): "al_p20_d6_w768_*_0.out",     (12, 384): "al_p20_d12_w384_*_0.out",
    (12, 768): "al_p20_d12_w768_*_0.out",   (12, 1152): "al_p20_d12_w1152_*_0.out",
    (12, 1536): "al_p20_d12_w1536_*_0.out", (18, 768): "al_p20_d18_w768_*_0.out",
    (24, 768): "al_p20_d24_w768_*_0.out",
}
P20_STEPS_PER_EPOCH = 152.5

# The 12 cells as they were fitted before the correction (kept only for the
# before/after comparison; do not fit new figures on these).
PREFIX_CELLS = {
    (6, 384): "fd_train_d6_w384_*_0.out",      (6, 768): "fd_train_d6_w768_*_0.out",
    (6, 1152): "fd_gridfill_d6_w1152_*_0.out", (6, 1536): "fd_gridfill_d6_w1536_*_0.out",
    (12, 384): "fd_train_d12_w384_*_0.out",    (12, 768): "fd_train_d12_w768_*_0.out",
    (12, 1152): "fd_widthext_d12_w1152_*_0.out", (12, 1536): "fd_widthext_d12_w1536_*_0.out",
    (18, 768): "fd_gridfill_d18_w768_*_0.out", (24, 768): "fd_gridfill_d24_w768_*_0.out",
    (48, 768): "fd_gridfill_d48_w768_*_0.out", (60, 768): "fd_gridfill_d60_w768_*_0.out",
}

# Ensemble replays at P=100M (init_shuffle, E in {2,3,4}, step resolution):
#   log-parsed from al_ensreplay_*.out ("[step S ens=E] val_loss=..."), cadence 304
# plus the pre-existing four cells in data_export/100M_data/ (cadence 152).
ENSEMBLE_REPLAY_LOGS = {
    (6, 1536): "al_ensreplay_d6_w1536_*_1.out",
    (24, 768): "al_ensreplay_d24_w768_*_1.out",
    (6, 768): "al_ensreplay_d6_w768_*_1.out",     # chain, pending
    (6, 384): "al_ensreplay_d6_w384_*_1.out",     # chain, pending
}
