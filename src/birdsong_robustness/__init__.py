"""birdsong-robustness: a controlled-degradation robustness benchmark for bioacoustic classifiers.

The package is deliberately split so that the *measurable* part and the *not-yet-measurable*
part never get confused with each other:

``degradations``
    Parameterised, physically-motivated audio degradations. Each one takes an explicit
    parameter (an SNR in dB, a distance in metres, a sample rate in Hz) and is required to
    produce exactly that effect.

``verify``
    Independent measurement of what a degradation actually did to a signal. Nothing in
    ``degradations`` is trusted; ``verify`` re-derives the effect from the output waveform.

``validate``
    The harness that sweeps every degradation across its parameter grid, measures the result
    with ``verify``, and prints a requested-vs-measured table. This is the part of the project
    that currently produces real numbers.

``protocol``
    The benchmark protocol: manifest schema, the canonical degradation grid, and the metric
    and harness code a classifier plugs into. The code is real and tested, but **no classifier
    has been evaluated yet** -- see the Results section of the README.

``noise`` / ``synth``
    Coloured- and environmental-noise models, and synthetic bird-song-like test signals. The
    repository ships no audio; every waveform used by the tests and the validation harness is
    generated at runtime from a seed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "degradations",
    "noise",
    "protocol",
    "synth",
    "validate",
    "verify",
]
