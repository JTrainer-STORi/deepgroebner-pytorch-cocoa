# Attribution Notice

This project is a derivative work of:

- `DeepGroebner`
- Upstream repository: https://github.com/dylanpeifer/deepgroebner

The upstream repository is identified on GitHub as using the GNU General Public License v3.0 (`GPL-3.0`).

Upstream work is associated with:

- Dylan Peifer
- Michael Stillman
- Daniel Halpern-Leistner

This repository contains substantial downstream modifications, including packaging changes, PyTorch instead of TensorFlow for the machine learning interface, a more versatile implementation of Buchberger in C++ utilizing CoCoALib for symboic algebra objects and functions, a more recent implementation of PPO with modifications like Popart value normalization, an expanded number of neural networks for working on this problem including attention-based and state conditioned neural networks, additional tutorials and more.


This repository is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

## Third-party dependency notice

This project depends on CoCoALib for the native commutative algebra backend.

- Project: CoCoALib
- Upstream: https://cocoa.dima.unige.it/cocoalib/
- License: GPL v3 or later

CoCoALib is not bundled in this repository; it is expected to be installed separately on the target system. See [INSTALL.md](INSTALL.md) for setup details.

When publishing this repository publicly, keep:

- the full GPL-3.0 license text
- this attribution notice
- prominent notices of modifications in changed files or release notes where appropriate
