import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        verdict: {
          safe: "#16a34a",
          caution: "#eab308",
          risk: "#dc2626",
        },
      },
    },
  },
  plugins: [],
};
export default config;
