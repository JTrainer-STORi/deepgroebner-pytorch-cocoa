# Installation

This project uses a compiled Cython/C++ extension backed by a system installation of CoCoALib.

## 1. Install system dependencies

You need:

- a C++17-capable compiler
- GMP
- CoCoALib Version 0.99880: available at https://cocoa.altervista.org/cocoalib/index.shtml
- Python 3.10+

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

Editable install:

```bash
pip install -e .
```

Regular install:

```bash
pip install .
```

## 4. Quick verification

Print the simplified training command without launching a run:

```bash
PYTHONPATH=src python -m deepgroebner_pytorch_cocoa.cli --dry-run True --track False
```

## Troubleshooting

If the build fails:

- confirm `COCOA_INCLUDE_DIR` points to the directory that contains the CoCoA headers
- confirm `COCOA_LIB_DIR` contains the CoCoALib library, or set `COCOA_LIBRARY_FILE`
- confirm GMP is installed and linkable on your system, and set `GMP_INCLUDE_DIR` / `GMP_LIB_DIR` if it is not in a standard location
- confirm your compiler supports `-std=c++17`
