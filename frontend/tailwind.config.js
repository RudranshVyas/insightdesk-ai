/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 900: "#0b0d10", 800: "#12151a", 700: "#1a1f27", 600: "#252b35" },
        line: "#2a3140",
      },
    },
  },
  plugins: [],
};
