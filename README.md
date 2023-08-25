# Odor Sampling (`odorsampling`)
Model of physicochemical odor sampling by animal olfactory systems

# Usage

## Installation

### Manual installation
1. [Clone the repository](https://github.com/cplab/odorsampling)
2. At the top folder structure (there should be a folder named `odorsampling` with a file named `layers.py` in it.), run the following command:
```sh
python -m odorsampling -h
```
This should print out a list of commands and arguments that can be configured.

### Linux users:
- *Remember to run* `sudo apt update && sudo apt upgrade` *and* `sudo apt-get update && sudo apt-get upgrade`
- You must install tkinter to display the graphs (if it isn't already installed). It will still run without it-
  however the only way to view the graphs would be through the generated PDFs.
- Run `sudo apt-get install python-tk` if you see an error that says something like "`Unable to load matplotlib backend [mpl_backend_name]. Please ensure any required packages are installed to use this backend.`"


## Cleanup
For convenience, a Makefile is provided with the following commands.

- `make clean`
  - This deletes all `.pdf`, `.csv`, and `.log` files. Useful for cleaning up between trials.

### Note:
This (`make`) requires Makefile to be installed.
- On Windows, it is recommended to install [`MSYS2`](https://www.msys2.org/) to get `make` for Windows.
- On Linux/Debian systems, this can be installed with the `apt` command, `sudo apt install make`. Many linux distributions come with it preinstalled.
- For Mac (untested), [use `xcode-select --install`](https://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac#answer-10265766). Also consider [this answer on the page](https://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac#answer-10265767) if you are having issues with `xcode-select`, something about needing to install it through the XCode GUI. Or, if you use Homebrew, `brew install make`, however some PATH setup may be required if using Homebrew.

## Git branches:
Checkout branches using `git checkout [branch_name]`
- `py3`
    - Python3 upgrade w/ data structures


