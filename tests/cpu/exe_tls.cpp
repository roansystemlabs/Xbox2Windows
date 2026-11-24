// CRT-free EXE whose own image carries implicit TLS data and two TLS callbacks.
//
// The EXE supplies the TLS directory the static C runtime would otherwise provide (_tls_used,
// _tls_index, the .tls start and end markers and the .CRT$XLA/.CRT$XLZ table bounds), so its template
// and callback table are the linker's own. The template holds g_tls (7) and the 8 KiB g_span array of
// tests/cpu/tls_value.cpp, so it spans at least one 4 KiB page boundary. Two callbacks, in .CRT$XLB
// (caller 1) and .CRT$XLM (caller 3), record each notification with observe(): g_tls in the low 16
// bits and the number of g_span words that differ from the template above them, 7 for a fresh block.
//
// exe_entry records 'E', writes 20 to the main thread's copy and runs two workers. Worker A writes 50
// and waits; worker B starts after A's write, writes 60 and waits. Then A is released and joined, and
// then B, so every notification arrives in a fixed order. The entry point prints one LOG line, an
// ordered list of events, and returns the main thread's final observe() value as the exit code, which
// the log repeats as its last event.
//
// Log events, separated by spaces:
//   <thread><reason><caller>=<value>  a TLS callback; thread m (initial), a or b; reason 0-3 as in
//                                     DLL_PROCESS_DETACH..DLL_THREAD_DETACH; value in hex
//   E                                 the entry point started
//   <thread>f=<v> <thread>w=<v> <thread>l=<v>  a worker's first, written and last observe() (m: main)
//   X=<exit code>
// Notifications from threads the EXE did not start (a host's own threads) are left out.
//
// exe_tls_mirror lets a host that maps the image give the callbacks four counters, one per
// notification reason, which it can still read after the image is released.
//
// Built with EARLY_THREAD (cpu_exe_tls_early), the first callback on process attach starts a probe
// thread p and waits for it; p records its own first observation. Under pe_map, p attaches while the
// process-attach callbacks run, before thread notifications are eligible, so the log must show pf but
// no notification for p. Under Windows, p's initialization waits for process initialization, so this
// variant is only run mapped.
//
// A failed setup ends the process with 0xE0000002 (CreateThread), 0xE0000003 (a wait did not
// complete) or 0xE0000006 (event log overflow).
#include <windows.h>
#include <intrin.h>

#include <cstdint>

#pragma runtime_checks("", off)

// The TLS support the C runtime's tlssup provides, for an EXE linked without it.
extern "C" {
ULONG _tls_index = 0;
#pragma data_seg(".tls")
char _tls_start = 0;
#pragma data_seg(".tls$ZZZ")
char _tls_end = 0;
#pragma data_seg()
#pragma section(".CRT$XLA", read)
#pragma section(".CRT$XLZ", read)
__declspec(allocate(".CRT$XLA")) PIMAGE_TLS_CALLBACK __xl_a = nullptr;
__declspec(allocate(".CRT$XLZ")) PIMAGE_TLS_CALLBACK __xl_z = nullptr;
extern const IMAGE_TLS_DIRECTORY64 _tls_used = {
    reinterpret_cast<ULONGLONG>(&_tls_start), reinterpret_cast<ULONGLONG>(&_tls_end),
    reinterpret_cast<ULONGLONG>(&_tls_index), reinterpret_cast<ULONGLONG>(&__xl_a + 1), 0, 0};
}

// The thread-locals below must be constant-initialized: this EXE has no C runtime to run a dynamic TLS
// initializer, so one would leave g_span zero in the template and fail to link (__dyn_tls_init). MSVC 19.42
// initializes a thread-local aggregate dynamically when its initializer calls a function, even a constexpr
// one, so g_span is spelled out with SPAN_WORD expressions instead of a constexpr call (#1094).
#define SPAN_WORD(i) (0xA5000000u ^ ((i) * 0x9E3779B1u))
#define SPAN_WORDS_4(i) SPAN_WORD(i), SPAN_WORD((i) + 1u), SPAN_WORD((i) + 2u), SPAN_WORD((i) + 3u)
#define SPAN_WORDS_16(i) SPAN_WORDS_4(i), SPAN_WORDS_4((i) + 4u), SPAN_WORDS_4((i) + 8u), SPAN_WORDS_4((i) + 12u)
#define SPAN_WORDS_64(i) \
    SPAN_WORDS_16(i), SPAN_WORDS_16((i) + 16u), SPAN_WORDS_16((i) + 32u), SPAN_WORDS_16((i) + 48u)
#define SPAN_WORDS_256(i) \
    SPAN_WORDS_64(i), SPAN_WORDS_64((i) + 64u), SPAN_WORDS_64((i) + 128u), SPAN_WORDS_64((i) + 192u)
#define SPAN_WORDS_1024(i) \
    SPAN_WORDS_256(i), SPAN_WORDS_256((i) + 256u), SPAN_WORDS_256((i) + 512u), SPAN_WORDS_256((i) + 768u)
#define SPAN_TEMPLATE {{SPAN_WORDS_1024(0u), SPAN_WORDS_1024(1024u)}}

namespace {
constexpr std::uint32_t SPAN_WORDS = 2048;  // 8 KiB, as in tests/cpu/tls_value.cpp

constexpr std::uint32_t span_word(std::uint32_t i) { return SPAN_WORD(i); }

struct Span {
    std::uint32_t words[SPAN_WORDS];
};

constexpr Span SPAN_CHECK = SPAN_TEMPLATE;  // the same list, checked word for word at compile time

constexpr bool span_template_complete() {
    for (std::uint32_t i = 0; i < SPAN_WORDS; ++i)
        if (SPAN_CHECK.words[i] != span_word(i)) return false;
    return true;
}
static_assert(span_template_complete(), "SPAN_TEMPLATE must list span_word(0..SPAN_WORDS-1) in order");
}  // namespace

__declspec(thread) int g_tls = 7;
__declspec(thread) Span g_span = SPAN_TEMPLATE;

namespace {
constexpr UINT CREATE_FAILED = 0xE0000002u, WAIT_FAILED_CODE = 0xE0000003u, LOG_FULL = 0xE0000006u;
constexpr DWORD WAIT_MS = 10000;  // shorter than the harness's process limit, so the exit code is observed

std::uint32_t span_changed() {
    std::uint32_t changed = 0;
    for (std::uint32_t i = 0; i < SPAN_WORDS; ++i) changed += g_span.words[i] != span_word(i);
    return changed;
}

DWORD observe() { return (span_changed() << 16) | static_cast<std::uint16_t>(g_tls); }

void write_copy(int value) {
    g_tls = value;
    g_span.words[0] ^= static_cast<std::uint32_t>(value);
    g_span.words[SPAN_WORDS - 1] ^= static_cast<std::uint32_t>(value);
}

struct Event {
    DWORD thread;
    char kind;  // '0'..'3' a notification reason, 'E' entry, 'f'/'w'/'l' a thread's own observation
    char caller;
    DWORD value;
};

Event g_events[64];
volatile LONG g_count;
DWORD g_threads[4];  // initial thread, worker A, worker B, probe; filled before each thread starts
volatile LONG* g_mirror;  // per-reason counters owned by a host, or null

void record(char kind, char caller, DWORD value) {
    const LONG index = InterlockedIncrement(&g_count) - 1;
    if (index >= static_cast<LONG>(sizeof(g_events) / sizeof(g_events[0]))) ExitProcess(LOG_FULL);
    g_events[index] = {GetCurrentThreadId(), kind, caller, value};
}

void mirror(DWORD reason) {
    if (g_mirror && reason < 4) InterlockedIncrement(&g_mirror[reason]);
}

#ifdef EARLY_THREAD
DWORD WINAPI probe(void*) {
    record('f', ' ', observe());
    return 0;
}

// Starts p from the first process-attach callback and waits until it has fully exited.
void run_probe() {
    HANDLE thread = CreateThread(nullptr, 0, probe, nullptr, CREATE_SUSPENDED, &g_threads[3]);
    if (!thread) ExitProcess(CREATE_FAILED);
    ResumeThread(thread);
    if (WaitForSingleObject(thread, WAIT_MS) != WAIT_OBJECT_0) ExitProcess(WAIT_FAILED_CODE);
    CloseHandle(thread);
}
#endif

void NTAPI callback_first(PVOID, DWORD reason, PVOID) {
    mirror(reason);
    record(static_cast<char>('0' + reason), '1', observe());
#ifdef EARLY_THREAD
    if (reason == DLL_PROCESS_ATTACH) run_probe();
#endif
}
void NTAPI callback_second(PVOID, DWORD reason, PVOID) {
    mirror(reason);
    record(static_cast<char>('0' + reason), '3', observe());
}
}  // namespace

extern "C" __declspec(dllexport) void exe_tls_mirror(volatile LONG* counts) { g_mirror = counts; }

#pragma section(".CRT$XLB", read)
#pragma section(".CRT$XLM", read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK p_exe_tls_first = callback_first;
extern "C" __declspec(allocate(".CRT$XLM")) PIMAGE_TLS_CALLBACK p_exe_tls_second = callback_second;

// The two callbacks in table order, exported like tls_value_callbacks in tests/cpu/tls_value.cpp.
extern "C" __declspec(dllexport) PIMAGE_TLS_CALLBACK tls_value_callbacks[2] = {callback_first, callback_second};

namespace {
struct Worker {
    HANDLE wrote;
    HANDLE resume;
    int value;
};

DWORD WINAPI worker(void* parameter) {
    auto* self = static_cast<Worker*>(parameter);
    record('f', ' ', observe());
    write_copy(self->value);
    record('w', ' ', observe());
    SetEvent(self->wrote);
    if (WaitForSingleObject(self->resume, WAIT_MS) != WAIT_OBJECT_0) ExitProcess(WAIT_FAILED_CODE);
    record('l', ' ', observe());
    return 0;
}

char g_line[64 * 24];
unsigned g_length;

void put(char c) { g_line[g_length++] = c; }

void put_hex(DWORD value) {
    static constexpr char hex[] = "0123456789abcdef";
    bool started = false;
    for (int shift = 28; shift >= 0; shift -= 4) {
        const unsigned digit = (value >> shift) & 15;
        if (digit || started || shift == 0) {
            put(hex[digit]);
            started = true;
        }
    }
}

char thread_name(DWORD thread) {
    for (int i = 0; i < 4; ++i)
        if (g_threads[i] == thread) return "mabp"[i];
    return 0;
}

void wait_or_exit(HANDLE handle) {
    if (WaitForSingleObject(handle, WAIT_MS) != WAIT_OBJECT_0) ExitProcess(WAIT_FAILED_CODE);
}
}  // namespace

extern "C" DWORD WINAPI exe_entry(void*) {
    g_threads[0] = GetCurrentThreadId();
    record('E', ' ', 0);
    record('f', ' ', observe());
    write_copy(20);
    record('w', ' ', observe());

    Worker workers[2] = {{CreateEventW(nullptr, TRUE, FALSE, nullptr), CreateEventW(nullptr, TRUE, FALSE, nullptr), 50},
                         {CreateEventW(nullptr, TRUE, FALSE, nullptr), CreateEventW(nullptr, TRUE, FALSE, nullptr), 60}};
    HANDLE threads[2];
    for (int i = 0; i < 2; ++i) {
        // Suspended, so the thread ID is known before the thread's own attach notification runs.
        threads[i] = CreateThread(nullptr, 0, worker, &workers[i], CREATE_SUSPENDED, &g_threads[1 + i]);
        if (!threads[i]) ExitProcess(CREATE_FAILED);
        ResumeThread(threads[i]);
        wait_or_exit(workers[i].wrote);  // B starts only after A has written, while A is still alive
    }
    for (int i = 0; i < 2; ++i) {  // A finishes and detaches before B resumes
        SetEvent(workers[i].resume);
        wait_or_exit(threads[i]);
        CloseHandle(threads[i]);
        CloseHandle(workers[i].wrote);
        CloseHandle(workers[i].resume);
    }
    const DWORD exit_code = observe();
    record('l', ' ', exit_code);

    put('L'), put('O'), put('G'), put(':');
    const LONG count = g_count;
    for (LONG i = 0; i < count; ++i) {
        const Event& event = g_events[i];
        const char name = thread_name(event.thread);
        if (!name) continue;  // a thread the EXE did not start
        if (event.kind == 'E') {
            put('E');
        } else {
            put(name);
            put(event.kind);
            if (event.caller != ' ') put(event.caller);
            put('=');
            put_hex(event.value);
        }
        put(' ');
    }
    put('X'), put('='), put_hex(exit_code), put('\n');
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), g_line, g_length, &written, nullptr);
    return exit_code;
}
