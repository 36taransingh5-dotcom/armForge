import type { Config } from "tailwindcss";

/** CSS vars hold "R G B" triplets so Tailwind's opacity modifiers
 * (bg-accent/10) work: they need `rgb(var(--x) / <alpha-value>)`, which a
 * plain hex custom property cannot supply an alpha channel for. */
const withOpacity = (variable: string) => `rgb(var(${variable}) / <alpha-value>)`;

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        bg: withOpacity("--bg"),
        fg: withOpacity("--fg"),
        muted: withOpacity("--muted"),
        line: withOpacity("--line"),
        accent: withOpacity("--accent"),
        warn: withOpacity("--warn"),
        "warn-bg": withOpacity("--warn-bg"),
        code: withOpacity("--code-bg"),
        ok: withOpacity("--ok"),
        bad: withOpacity("--bad"),
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
