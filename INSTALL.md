# Installation

This project uses a compiled Cython/C++ extension backed by a system installation of CoCoALib.

## 1. Install system dependencies

You need:

- a C++17-capable compiler
- GMP Version 5.1.0 or newer: https://gmplib.org/
- CoCoALib Version 0.99880: available at https://cocoa.altervista.org/cocoalib/index.shtml
- Python 3.10+

CoCoALib requires GMP as a separate system dependency. This project has been tested with modern system GMP installs, and CoCoALib's local configure script requires at least GMP `5.1.0`.

Typical GMP installation commands:

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgmp-dev
```

macOS with Homebrew:

```bash
brew install gmp
```

If you build GMP yourself or install it into a custom prefix, you may need to set `GMP_INCLUDE_DIR` and `GMP_LIB_DIR` explicitly later in the install process.

## 2. Tell the build where CoCoALib lives

If CoCoALib is installed in one of the following locations, the build will find it automatically:

The build currently checks these include directories:

- `/usr/local/include`
- `/usr/include`
- `/opt/homebrew/include`
- `/opt/homebrew/opt/gmp/include`
- `/usr/local/opt/gmp/include`
- `/opt/local/include`

and these library directories:

- `/usr/local/lib`
- `/usr/lib`
- `/usr/lib/x86_64-linux-gnu`
- `/opt/homebrew/lib`
- `/opt/homebrew/opt/gmp/lib`
- `/usr/local/opt/gmp/lib`
- `/opt/local/lib`

On a typical Linux machine, the most common "standard" install is:

- headers under `/usr/local/include/CoCoA/`
- library at `/usr/local/lib/libcocoa.a` or `/usr/local/lib/libcocoa.so`

Important: building CoCoALib in a local checkout directory does not automatically make it a "standard system install" for this package. For example, if you downloaded and built CoCoALib under a directory such as `~/CoCoALib` or `~/deepgroebner-pytorch-cocoa/CoCoALib`, the headers and library will usually still live in that checkout's `include/` and `lib/` directories unless you explicitly install or copy them into a standard location such as `/usr/local/include` and `/usr/local/lib`.

So if you only built CoCoALib locally and did not install or copy its files into a standard system location, you should set the environment variables below instead of assuming the build will find it automatically.

If it is installed somewhere custom, set:

```bash
export COCOA_INCLUDE_DIR=/path/to/include
export COCOA_LIB_DIR=/path/to/lib
```

If you prefer to link against a specific CoCoALib archive or shared object directly, set:

```bash
export COCOA_INCLUDE_DIR=/path/to/include
export COCOA_LIBRARY_FILE=/path/to/libcocoa.a
```

If GMP is installed somewhere non-standard, you can point the build at it explicitly too:

```bash
export GMP_INCLUDE_DIR=/path/to/gmp/include
export GMP_LIB_DIR=/path/to/gmp/lib
```

Optional override for the library name when using `COCOA_LIB_DIR`:

```bash
export COCOA_LIBRARY=cocoa
```

## 3. Install the Python package

Run the following commands from the `deepgroebner-pytorch-cocoa` repository root.

Editable install:

```bash
pip install -e .
```

Use this if you are developing inside this checkout. The package stays linked to the source tree, so local code edits are picked up without reinstalling.

Regular install:

```bash
pip install .
```

Use this if you just want a normal install of the current checkout into your Python environment. If you later change the source files, you will need to reinstall by running this command again for those changes to take effect.

## 4. Quick verification

Run the installed verification command:

```bash
deepgroebner-pytorch-cocoa-verify
```

If the install is working, this command should:

- import the Python package
- import the compiled `wrapped` extension
- reset and step both supported environments once
- print one long `python -m deepgroebner_pytorch_cocoa.ppo_masked ...` training command
- exit without a traceback or import error

A successful output will look roughly like:

```text
OK package import: deepgroebner_pytorch_cocoa
OK compiled extension import: deepgroebner_pytorch_cocoa.wrapped
OK BuchbergerExponents-v1 reset/step: reward=-4.0 terminated=False truncated=False
OK BuchbergerExponentsNoPQ-v1 reset/step: reward=-4.0 terminated=False truncated=False
OK public train command: python -m deepgroebner_pytorch_cocoa.ppo_masked --env.gym_id=BuchbergerExponents-v1 ...
Installation verification succeeded.
```

The exact command line will be longer than this example, but the important thing is that all checks print `OK ...` lines and the final line is `Installation verification succeeded.`

```

## Troubleshooting

If the build fails:

- confirm `COCOA_INCLUDE_DIR` points to the directory that contains the CoCoA headers
- confirm `COCOA_LIB_DIR` contains the CoCoALib library, or set `COCOA_LIBRARY_FILE`
- confirm GMP is installed and linkable on your system, and set `GMP_INCLUDE_DIR` / `GMP_LIB_DIR` if it is not in a standard location
- confirm your compiler supports `-std=c++17`
