#!/bin/bash

# Set default values
venv_name="chord"
venv_path="./venv"
code_path="./venv/src"
ignore_system="false"
reset_env="false"
legacy_setup="false"

help_message="$(basename $0) [-n VENV_NAME] [-v VENV_PATH] [-e CODE_PATH] [-i] [-r] \n\n
                -n  \t Virtual environment name that will appear in your terminal prompt when activated. \n
                    \t\t Defaults to '$venv_name'. \n  
                -v  \t Path where virtual environment will be installed. \n
                    \t\t Defaults to '$venv_path'. \n
                -e  \t Path where packages will be cloned for development. \n
                    \t\t Defaults to '$code_path'. \n
                -i  \t Ignore packages already installed on your system, i.e., install all dependencies. \n
                -r  \t Remove any existing virtual environment at VENV_PATH before creating a new one. \n"

# Parse any options provided by user
while getopts 'hn:v:e:irl' OPTION; do
    case "$OPTION" in
        n)
            venv_name="($OPTARG)"
            ;;

        v)
            venv_path="$OPTARG"
            ;;

        e)
            code_path="$OPTARG"
            ;;

        i)
            ignore_system="true"
            ;;

        r)
            reset_env="true"
            ;;

        h)
            echo -e $help_message >&2
            exit 1
            ;;

        ?)
            echo -e $help_message >&2
            exit 1
            ;;
    esac
done
shift "$(($OPTIND -1))"

# Optionally remove existing environment
if ${reset_env} && [ -d "$venv_path" ]; then
    rm -rf "$venv_path"
fi

# Create the virtual environment
if ${ignore_system}; then
    virtualenv --prompt=$venv_name $venv_path
else
    virtualenv --system-site-packages --prompt=$venv_name $venv_path
fi
source $venv_path/bin/activate

# Install the packages
mkdir -p $code_path
pip install --use-deprecated=legacy-resolver Cython
if ${ignore_system}; then
    pip install --use-deprecated=legacy-resolver --src $code_path -r requirements.txt
else
    pip install --no-build-isolation --use-deprecated=legacy-resolver --src $code_path -r requirements.txt
fi

# Install this package (modern by default; legacy option retained for compatibility)
if ${ignore_system}; then
    pip install --use-deprecated=legacy-resolver -e .
else
    pip install --no-build-isolation --use-deprecated=legacy-resolver -e .
fi
