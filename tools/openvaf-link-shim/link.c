/* MSVC link.exe shim for OpenVAF on machines without Visual Studio C++ tools.
 * Translates "link /NOLOGO objs... /OUT:x /DLL msvcrt.lib import.lib" into
 * "gcc -shared -o x objs... import.lib" using MinGW gcc. */
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    char cmd[65536] = "gcc -shared -static-libgcc";
    const char *out = NULL;
    /* chkstk.o lives next to this executable. */
    char shim_dir[4096];
    GetModuleFileNameA(NULL, shim_dir, sizeof shim_dir);
    for (char *c = shim_dir; *c; c++) if (*c == '\\') *c = '/';
    char *slash = strrchr(shim_dir, '/');
    if (slash) *slash = 0; else strcpy(shim_dir, ".");
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strncmp(a, "/OUT:", 5) == 0) { out = a + 5; continue; }
        if (a[0] == '/') continue;                       /* /NOLOGO, /DLL, ... */
        if (strcmp(a, "msvcrt.lib") == 0) continue;      /* MinGW links msvcrt by default */
        strcat(cmd, " \""); strcat(cmd, a); strcat(cmd, "\"");
    }
    if (!out) { fprintf(stderr, "link shim: no /OUT: given\n"); return 1; }
    strcat(cmd, " \""); strcat(cmd, shim_dir); strcat(cmd, "/chkstk.o\" -lucrtbase -o \""); strcat(cmd, out); strcat(cmd, "\"");
    if (getenv("LINK_SHIM_VERBOSE")) fprintf(stderr, "%s\n", cmd);
    return system(cmd);
}
