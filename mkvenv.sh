#!/bin/bash

# Set default values
venv_name="chord"
venv_path="./venv"
code_path="./venv/src"
ignore_system="false"

help_message="$(basename $0) [-p VENV_NAME] [-v VENV_PATH] [-e CODE_PATH] [-i] \n\n
                -n  \t Virtual environment name that will appear in your terminal prompt when activated. \n
                    \t\t Defaults to '$venv_name'. \n  
                -v  \t Path where virtual environment will be installed. \n
                    \t\t Defaults to '$venv_path'. \n
                -c  \t Path where packages will be cloned for development. \n
                    \t\t Defaults to '$code_path'. \n
                -i  \t Ignore packages already installed on your system, i.e., install all dependencies. \n"

# Parse any options provided by user
while getopts 'hp:v:e:s' OPTION; do
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

# Create the virtual environment
if ${ignore_system}; then
    virtualenv --prompt=$venv_name $venv_path
else
    virtualenv --system-site-packages --prompt=$venv_name $venv_path
fi
source $venv_path/bin/activate

# Install the packages
mkdir $code_path
if ${ignore_system}; then
    pip install --use-deprecated=legacy-resolver --src $code_path -r requirements.txt
else
    pip install --no-build-isolation --use-deprecated=legacy-resolver --src $code_path -r requirements.txt
fi
python setup.py develop