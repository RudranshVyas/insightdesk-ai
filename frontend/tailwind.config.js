/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm paper, not the cream-and-serif cliche: cooler, closer to stone.
        paper: { DEFAULT: "#FAFAF8", raised: "#FFFFFF", sunk: "#F3F2EE" },
        ink: { DEFAULT: "#16181C", soft: "#454A54", mute: "#767C88", faint: "#A6ABB5" },
        rule: { DEFAULT: "#E4E2DC", strong: "#D2CFC7" },
        // One accent, used sparingly. Institutional teal — calm, not corporate blue.
        teal: { DEFAULT: "#0E5049", deep: "#093733", light: "#E6EFED", mid: "#2C7A70" },
        // Status colours are deliberately desaturated so the page stays calm.
        good: { DEFAULT: "#2F6A4F", bg: "#EAF2EC" },
        caution: { DEFAULT: "#8A5A18", bg: "#FBF1E2" },
        halt: { DEFAULT: "#8B3A3A", bg: "#F9EBEB" },
      },
      fontFamily: {
        serif: ['"IBM Plex Serif"', "Georgia", "serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      maxWidth: { reading: "44rem" },
      boxShadow: {
        card: "0 1px 2px rgba(22,24,28,.04), 0 8px 24px -12px rgba(22,24,28,.10)",
        lift: "0 2px 4px rgba(22,24,28,.05), 0 16px 40px -16px rgba(22,24,28,.16)",
      },
      keyframes: {
        rise: { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "none" } },
        sweep: { from: { transform: "scaleX(0)" }, to: { transform: "scaleX(1)" } },
      },
      animation: {
        rise: "rise .5s cubic-bezier(.2,.7,.3,1) both",
        sweep: "sweep .8s cubic-bezier(.2,.7,.3,1) both",
      },
    },
  },
  plugins: [],
};
