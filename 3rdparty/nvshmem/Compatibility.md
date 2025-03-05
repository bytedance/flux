# Compatibility with NVSHMEM

NVSHMEM follows semantic versioning for its releases and packages per commit i.e `MAJOR.MINOR.PATCH.TWEAK`.
- Each component of the version is monotonically increasing number. So, if the author makes non-source change e.g. `test`, `perftest`, etc, it would require updating `TWEAK` component of the version.
- If the author makes a change to the source file, but not the ABI or API, it is PATCH change by 1 and `TWEAK` resets.
- If the author makes a change to the API/ABI definition in a backward compat way, it is MINOR change by 1 and TWEAK/PATCH reset to 0.
- If the author makes a change to the ABI/API definition in the non-backward compat way, it is MAJOR change by 1 and TWEAK/PATCH/MINOR resets to 0.
