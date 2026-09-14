# OpenVAF link shim for Windows

OpenVAF-reloaded's Windows binary compiles Verilog-A to COFF objects and then
runs MSVC's `link.exe` (`link /NOLOGO objs /OUT:x.osdi /DLL msvcrt.lib
__openvaf__import.lib`). On a machine without the Visual Studio C++ workload
that step fails. This shim is a drop-in `link.exe` that rewrites the command
into `gcc -shared` for a 64-bit MinGW toolchain.

Two things MinGW does not provide for MSVC-targeted objects are supplied here:

- `__chkstk` (MSVC's stack probe, emitted for PSP103's large frames) is aliased
  to libgcc's `___chkstk_ms`, which has the same contract, in `chkstk.S`.
- `__stdio_common_vsprintf` from the Universal CRT comes from `-lucrtbase`.

## Build

Install a 64-bit MinGW such as [w64devkit](https://github.com/skeeto/w64devkit)
(portable, no admin rights) and run:

```sh
PATH=/c/Users/<you>/tools/w64devkit/bin:$PATH sh build.sh
```

## Compile the IHP models

Sparse-clone `IHP-Open-PDK` (only `ihp-sg13g2/libs.tech/verilog-a` and
`ihp-sg13g2/libs.tech/ngspice` are needed), then with the shim directory
*first* on PATH so it shadows both MSVC `link` and coreutils `link`:

```sh
cd <pdk>/ihp-sg13g2/libs.tech/verilog-a
export PATH=<repo>/tools/openvaf-link-shim:<w64devkit>/bin:$PATH
mkdir -p ../ngspice/osdi
for m in psp103/psp103 psp103/psp103_nqs mosvar/mosvar; do
  openvaf-r -D__NGSPICE__ -o ../ngspice/osdi/$(basename $m).osdi $m.va
done
```

Set `IHP_PDK_ROOT=<pdk>` and run `python scripts/check_pdk.py`. Windows
ngspice 47 loads the resulting `.osdi` files directly; the AC gate and HD3 for
the submission design reproduce the Linux numbers exactly.

`LINK_SHIM_VERBOSE=1` prints the gcc command the shim runs.
