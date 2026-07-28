"""Import tensorflow before xgboost at session start.

On this platform, importing xgboost (and its bundled OpenMP runtime) before
tensorflow causes tensorflow's threading initialization to deadlock the first
time a Keras model is trained later in the same process. Importing tensorflow
first, before pytest collects any test module, avoids the conflict regardless
of what individual test files import in what order.
"""

import tensorflow  # noqa: F401
