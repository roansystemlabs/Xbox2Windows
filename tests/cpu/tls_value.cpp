// A DLL with implicit thread-local storage read inside its TLS callbacks.
//
// g_tls lives in the PE TLS data template (non-empty), unlike tls_route.asm
// whose template is empty. g_span adds 8 KiB of initialized words to the same
// template, so the template spans at least one 4 KiB page boundary and every
// word on either side of it holds a distinct value (span_word below).
//
// Two TLS callbacks are registered, in .CRT$XLB (caller tag 1) and .CRT$XLM
// (caller tag 3), so the linker places them in that order in the callback
// table. Each callback and DllMain (caller tag 2, as in tls_route) forwards
// observe() in the sink's reserved slot: g_tls in the low 16 bits and the number
// of g_span words that differ from the template in the bits above. A thread
// whose block was copied from the template therefore reports 7. The sink keeps
// the calls in order, so the harness sees the table order on every thread.
//
// tls_value_callbacks exports the two callback addresses in table order so the
// harness can find them in the TLS callback table of the built image.
//
// Built with the static MSVC runtime like the life_* fixtures, so the linker
// emits a real TLS directory with data. pe_map serves the template through a
// carrier DLL that Windows loads (tools/pe_map.cpp), so both loaders must report
// the same values. See Refs #165 and #870.
#include <windows.h>

#include <cstdint>

extern "C" __declspec(dllimport) void sink_record(void* module, unsigned reason, void* reserved,
                                                  unsigned caller);

constexpr std::uint32_t SPAN_WORDS = 2048;  // 8 KiB

// g_span must be constant-initialized so its words are in the TLS data template; a dynamic
// initializer would leave the template zero and fill each thread's copy later, through the C
// runtime. MSVC 19.42 initializes a thread-local aggregate dynamically when its initializer calls a
// function, even a constexpr one, so g_span is spelled out with SPAN_WORD expressions (#1094).
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

// The template value of g_span[i]; tests/cpu/native_tls_value_check.py computes the same words.
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

// Global scope by design: the linker must emit g_tls and g_span in the TLS data
// template and the callback addresses in .CRT$XL*. An anonymous namespace would
// give them internal linkage and obscure the template in a dumpbin review.
__declspec(thread) int g_tls = 7;
__declspec(thread) Span g_span = SPAN_TEMPLATE;

constexpr unsigned CALLER_TLS = 1;
constexpr unsigned CALLER_ENTRY = 2;
constexpr unsigned CALLER_TLS_SECOND = 3;

// Number of g_span words on this thread that differ from the template.
static std::uint32_t span_changed() {
    std::uint32_t changed = 0;
    for (std::uint32_t i = 0; i < SPAN_WORDS; ++i) changed += g_span.words[i] != span_word(i);
    return changed;
}

// g_tls in the low 16 bits, the changed g_span words above: 7 for a block copied from the template.
static std::uintptr_t observe() {
    return (static_cast<std::uintptr_t>(span_changed()) << 16) | static_cast<std::uint16_t>(g_tls);
}

void NTAPI tls_callback(void* module, unsigned long reason, void* reserved) {
    (void)reserved;  // the observed TLS state travels in the sink's reserved slot by design
    sink_record(module, reason, reinterpret_cast<void*>(observe()), CALLER_TLS);
}

void NTAPI tls_callback_second(void* module, unsigned long reason, void* reserved) {
    (void)reserved;
    sink_record(module, reason, reinterpret_cast<void*>(observe()), CALLER_TLS_SECOND);
}

#pragma section(".CRT$XLB", read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK p_tls_value_callback = tls_callback;
#pragma section(".CRT$XLM", read)
extern "C" __declspec(allocate(".CRT$XLM")) PIMAGE_TLS_CALLBACK p_tls_value_callback_second = tls_callback_second;

extern "C" __declspec(dllexport) PIMAGE_TLS_CALLBACK tls_value_callbacks[2] = {tls_callback, tls_callback_second};

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH || reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH ||
        reason == DLL_PROCESS_DETACH) {
        sink_record(module, reason, reinterpret_cast<void*>(observe()), CALLER_ENTRY);
    }
    return TRUE;
}

extern "C" __declspec(dllexport) int tls_value_read() { return g_tls; }

extern "C" __declspec(dllexport) void tls_value_set(int value) { g_tls = value; }

// The calling thread's observe() value.
extern "C" __declspec(dllexport) std::uint64_t tls_value_observe() { return observe(); }

// Changes this thread's copy only: g_tls and the first and last g_span words, which lie on
// different pages of the template.
extern "C" __declspec(dllexport) void tls_value_write(int value) {
    g_tls = value;
    g_span.words[0] ^= static_cast<std::uint32_t>(value);
    g_span.words[SPAN_WORDS - 1] ^= static_cast<std::uint32_t>(value);
}

// A worker that stays alive while another worker writes. It records observe() when it starts,
// writes `value` to its own copy, records again, signals `wrote`, waits for `resume` and records a
// third time, so the harness can check that another thread's write did not reach this copy.
struct Worker {
    HANDLE wrote;
    HANDLE resume;
    std::int32_t value;
    std::int32_t waited;  // WaitForSingleObject result for resume
    std::uint64_t first;
    std::uint64_t written;
    std::uint64_t last;
};

extern "C" __declspec(dllexport) DWORD WINAPI tls_value_worker(LPVOID parameter) {
    auto* worker = static_cast<Worker*>(parameter);
    worker->first = observe();
    tls_value_write(worker->value);
    worker->written = observe();
    SetEvent(worker->wrote);
    worker->waited = static_cast<std::int32_t>(WaitForSingleObject(worker->resume, 10000));
    worker->last = observe();
    return 0;
}
