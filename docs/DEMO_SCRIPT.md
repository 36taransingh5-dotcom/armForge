# Demo script

~2:40, under the 3:00 cap. Every number below is real and pulled from
[docs/RESULTS.md](RESULTS.md) — read it once before recording so you can
adlib if you fumble a line; you know these numbers, they're not memorized
trivia.

Record with QuickTime (macOS: File → New Screen Recording) at 1280×800 or
larger in the browser window. Narrate live as you click — don't record
silent then dub over, the timing won't match.

**Before you hit record:**
- Backend running: `PYTHONPATH=backend uvicorn app.main:app --port 8000`
- Frontend running: `cd frontend && npm run dev`, open `localhost:3000`
- Close other tabs/apps, silence notifications
- Have the Neoverse-N2 historical run pre-loaded in a second tab so you're not waiting on a click
- Do one full dry run first. Second take is always better.

---

## 0:00–0:20 — The problem

**[Show: terminal, or just talk to camera/screen]**

> "Deploying an LLM on Arm means guessing at quantization, thread count, and
> runtime — and the guesses are usually wrong, because the tools that would
> tell you don't exist. ArmForge reads what an Arm CPU can actually do, and
> proves — with real benchmarks — what configuration wins on that specific
> chip."

## 0:20–0:45 — Hardware, not a lookup table

**[Show: dashboard at localhost:3000, hardware card]**

> "This is my Apple M4 — ArmForge just detected it has 4 performance cores
> and 6 efficiency cores, and a matrix engine called SME2. That detail
> matters in about ninety seconds."

**[Point at the feature tags: i8mm, sme2, dotprod]**

> "It's not just reading 'Arm64' — it's reading the actual instruction set
> extensions this chip implements, because different Arm chips implement
> different subsets."

## 0:45–1:30 — Run it live

**[Click "New optimization." Select Q4_0 model, cpu runtime, long-context workload.]**

> "I'll kick off a real sweep — this is running actual inference through
> llama.cpp, not simulated."

**[Click "Run sweep." Show the live progress bar advancing.]**

> "ArmForge doesn't guess a config and stop — it builds a measurement plan
> from what the CPU reported, then benchmarks every candidate for real."

**[Cut/speed up here if the sweep takes a while — or switch to a pre-completed run and say so on camera: "here's one I ran earlier so we're not waiting."]**

**[Show the finished recommendation card]**

> "And here's the result: prompt processing peaks at 6 threads, token
> generation peaks at 2. Those are different numbers on purpose —
> ArmForge configures them separately, because they're bound by different
> resources. If you'd used all 10 cores — what `nproc` tells you to do —
> decode drops to 3.4 tokens per second. Tuned, it's 99. That's a 29x gap,
> and it's specific to this chip's heterogeneous cores."

## 1:30–2:00 — It's not tuned to one chip

**[Switch tab: historical run, neoverse-n2-sweep-longcontext]**

> "Same tool, different Arm chip — this is an AWS Graviton-class Neoverse-N2,
> benchmarked for real on GitHub's Arm64 CI runners, not simulated. Four
> uniform cores, no efficiency-core split, no SME — it has SVE instead."

**[Point at the recommendation: -t 4 -tb 4]**

> "The recommendation collapses to one number here — 4 and 4 — because on
> this chip prefill and decode agree. That's the right answer changing with
> the hardware, not a hardcoded default."

**[Optional if time: point at the KleidiAI section of RESULTS.md or mention verbally]**

> "We even caught Arm's own KleidiAI kernels only helping when the chip has
> the matrix engine to use them — on this one, enabling them changes nothing
> measurable. On the M4, it's up to 66% faster single-threaded."

## 2:00–2:30 — Deploy it

**[Scroll to "Download deployment package," click it, show the zip / or show run.sh in a terminal]**

> "Every run exports a real deployment package — a run script with the exact
> thread split baked in, a Dockerfile pinned to the exact llama.cpp build,
> and a report. This isn't a recommendation you have to remember — it's a
> command you run."

**[Optional: terminal, run the exported run.sh from the M4 sweep, show `n_threads = 2 (n_threads_batch = 6)` in llama.cpp's own output — must match the -t 2 -tb 6 shown on screen a moment earlier, so run the package from the same sweep you just demoed, not a different one]**

> "And that's llama.cpp itself confirming the split actually took effect —
> not just written to a file."

## 2:30–2:40 — Close

**[Back to dashboard or just camera]**

> "ArmForge turns Arm inference optimization from guesswork into something
> you can measure, trust, and ship. Thanks."

---

## If you're short on time, cut in this order

1. Cut the Dockerfile/terminal confirmation beat (2:00–2:30) down to just showing the zip download.
2. Cut the KleidiAI aside in 1:30–2:00.
3. Never cut the cross-Arm comparison (1:30–2:00 core) — it's the single
   strongest piece of evidence that this isn't a lookup table, and it's
   worth more than any other 30 seconds you could spend.

## Numbers to have memorized cold (say them without reading)

- **29× decode gap** at nproc vs tuned, on the M4
- **`-t 2 -tb 6`** M4, **`-t 4 -tb 4`** Neoverse-N2 — the split that changes with hardware
- **2.28×** faster prompt processing, Q4_0 vs Q4_K_M, on the M4
