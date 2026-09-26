# The device architectures the registered card profiles describe
# (crates/turbine-kernels/src/cards/*.rs, `CardProfile::archs`). libturbine_hip.so is built for
# these by default and refuses any other GPU_TARGETS entry. Adding a card family = a profile in
# cards/ plus its architecture here; turbine-kernels `cards::tests::cmake_lists_every_profile_arch`
# fails while the two lists differ.
set(TURBINE_PROFILE_ARCHS gfx1201)
