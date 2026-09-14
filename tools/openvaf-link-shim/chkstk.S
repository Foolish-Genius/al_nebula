/* MSVC __chkstk has the same contract as libgcc's ___chkstk_ms on x64:
 * rax = bytes to probe, all registers preserved, rsp untouched. */
    .text
    .globl __chkstk
__chkstk:
    jmp ___chkstk_ms
