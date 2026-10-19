# Installation


In this package, the Buchberger environment is coded in C++ using classes and functions from the CoCoALib library so it is necessary to install CoCoALib prior to installing deepgroebner-pytorch-cocoa

## 1. Install system dependencies

You need:

- a C++17-capable compiler
- GMP Version 5.1.0 or newer: https://gmplib.org/
- CoCoALib Version 0.99880: available at https://cocoa.altervista.org/cocoalib/index.shtml
- Python 3.10+


Typical GMP installation commands:

Ubuntu:

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

On a typical Linux machine, the most common installation location is:

- headers under `/usr/local/include/CoCoA/`
- library at `/usr/local/lib/libcocoa.a` or `/usr/local/lib/libcocoa.so`

Important: building CoCoALib in a local checkout directory does not automatically make it a "standard system install" for this package. For example, if you downloaded and built CoCoALib under a directory such as `~/CoCoALib` or `~/deepgroebner-pytorch-cocoa/CoCoALib`, the headers and library will usually still live in that checkout's `include/` and `lib/` directories unless you explicitly install or copy them into a standard location such as `/usr/local/include` and `/usr/local/lib`.

So if you only built CoCoALib locally and did not install or copy its files into a standard system location, you should set the environment variables below instead of assuming the build will find it automatically.

If it is installed somewhere custom, set:

```bash
export COCOA_INCLUDE_DIR=/path/to/include
export COCOA_LIB_DIR=/path/to/lib
```

If GMP is installed somewhere non-standard, you can point the build at it explicitly too:

```bash
export GMP_INCLUDE_DIR=/path/to/gmp/include
export GMP_LIB_DIR=/path/to/gmp/lib
```

## 3. Install the Python package

Run the following commands from the `deepgroebner-pytorch-cocoa` repository root.

Editable install:

```bash
pip install -e .
```

Use this if you plan to make edits to some of the python source code and do not want to have to reinstall the library every time you do.

Cython or C++ source changes still require rebuilding the extension by rerunning the installation command.

Regular install:

```bash
pip install .
```

Use this if you just want a normal install of the current checkout into your Python environment. If you later change the python source files, you will need to reinstall by running this command again for those changes to take effect.

By default the installation will not install the Weights & Biases library, to include it in the install use:

```bash
pip install ".[tracking]"
```


## 4. Quick verification

Run the installed verification command from any directory:

```bash
deepgroebner-pytorch-cocoa-verify
```

It imports the package and compiled CoCoALib extension, resets and steps both
registered environments on a small fixed ideal, and runs the default PyTorch
actor and critic on CPU for each environment. It requires only the standard
package dependencies and does not start training or create run artifacts.
Successful checks print `OK ...` lines followed by
`Installation verification succeeded.`; a failed check exits with a nonzero status.
Use `--verbosity quiet` to print only the success message.

## 5. Using the library

Once the installation is complete, you are ready to start using the package. Look at the [readme](README.md)

## Troubleshooting

If the build fails:

- confirm `COCOA_INCLUDE_DIR` points to the directory that contains the CoCoA headers
- confirm `COCOA_LIB_DIR` contains the CoCoALib library, or set `COCOA_LIBRARY_FILE`
- confirm GMP is installed and linkable on your system, and set `GMP_INCLUDE_DIR` / `GMP_LIB_DIR` if it is not in a standard location
- confirm your compiler supports `-std=c++17`
