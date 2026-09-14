#!/bin/sh
# Build the link.exe shim with a 64-bit MinGW gcc (e.g. w64devkit) on PATH.
set -e
cd "$(dirname "$0")"
gcc -c -o chkstk.o chkstk.S
gcc -O2 -o link.exe link.c
echo "built $(pwd)/link.exe"
