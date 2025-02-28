#!/bin/bash

base='--cached'
target='master'
show_only=0
fail_on_diff=0
format_all=0

while [ "$#" -gt 0 ]; do
  case $1 in
  -h | --help)
    echo "Usage: $0 [--fail-on-diff] [--show-only] [--format-all] [--base|-b <base(=--cached)>] [--target|-t <target(=origin/master)>]"
    exit 0
    ;;
  --fail-on-diff)
    fail_on_diff=1
    shift
    ;;
  --show-only)
    show_only=1
    shift
    ;;
  --format-all)
    format_all=1
    shift
    ;;
  --base | -b)
    base="$2"
    shift
    shift
    ;;
  --target | -t)
    target="$2"
    shift
    shift
    ;;
  *)
    echo "error: unrecognized option $1"
    exit 1
    ;;
  esac
done

# hack for code format
if [ -f "pre-commit" ]; then
  cp pre-commit ./.git/hooks/pre-commit
fi

files_to_check=()
if [ "$format_all" -eq 1 ]; then
  files_to_check=($(git ls-files --exclude-standard | grep -v "$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }' | xargs -I {} echo -n '\|{}' | cut -c 3-)"))
else
  files_to_check=($(git diff $base $target --name-only --diff-filter=ACMRT))
fi

files_to_check_cpp=$(printf "%s\n" "${files_to_check[@]}" | grep "\.\(c\|cpp\|cc\|h\|hpp\|cu\|cuh\)$")
files_to_check_py=$(printf "%s\n" "${files_to_check[@]}" | grep "\.\(py\)$")
files_to_check_pyi=$(printf "%s\n" "${files_to_check[@]}" | grep "\.\(pyi\)$")

if [ "$files_to_check_cpp" = "" ] && [ "$files_to_check_py" = "" ] && [ "$files_to_check_pyi" = "" ]; then
  exit 0
fi

if ! clang-format -version >/dev/null; then
  echo "error: clang-format not found, can be installed by: pip3 install clang-format"
  exit 1
fi

if ! black --version >/dev/null; then
  echo "error: black not found, can be installed by: pip3 install black"
  exit 1
fi

if [ "$show_only" -eq 1 ]; then
  # cpp files
  for f in $files_to_check_cpp; do
    if [ -L $f ]; then
      continue
    fi
    result=$(diff <(git show :$f) <(git show :$f | clang-format -style=file --assume-filename=$f))
    if [ "$result" != "" ]; then
      echo "===== $f ====="
      echo -e "$result"
      has_diff=1
    fi
  done
  # .py files
  for f in $files_to_check_py; do
    if [ -L $f ]; then
      continue
    fi
    result=$(diff <(git show :$f) <(git show :$f | black -q -))
    if [ "$result" != "" ]; then
      echo "===== $f ====="
      echo -e "$result"
      has_diff=1
    fi
  done
  # .pyi files
  for f in $files_to_check_pyi; do
    if [ -L $f ]; then
      continue
    fi
    result=$(diff <(git show :$f) <(git show :$f | black --pyi -q -))
    if [ "$result" != "" ]; then
      echo "===== $f ====="
      echo -e "$result"
      has_diff=1
    fi
  done
else
  echo "Formatting cpp files by clang-format..."
  echo $files_to_check_cpp | xargs clang-format -i --style=file
  if [[ ! -z $files_to_check_py ]]; then
    echo "Formatting py files by black..."
    echo $files_to_check_py | xargs black -q
  fi
  if [[ ! -z $files_to_check_pyi ]]; then
    echo "Formatting pyi files by black..."
    echo $files_to_check_pyi | xargs black --pyi -q
  fi
fi

if [[ "$fail_on_diff" -eq 1 ]] && [[ "$has_diff" -eq 1 ]]; then
  echo "code format check failed, please run the following command before commit: ./code-format.sh"
  exit 1
fi
