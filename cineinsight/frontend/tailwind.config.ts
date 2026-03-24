import type { Config } from "tailwindcss";
import { heroui } from "@heroui/react";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@heroui/**/theme/dist/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        netflix: {
          black: "#141414",
          dark: "#1a1a1a",
          card: "#222222",
          red: "#E50914",
          "red-hover": "#F40612",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "Helvetica Neue",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  darkMode: "class",
  plugins: [
    heroui({
      defaultTheme: "dark",
      themes: {
        dark: {
          colors: {
            background: "#141414",
            foreground: "#ECEDEE",
            primary: {
              50: "#FFF1F2",
              100: "#FFE4E6",
              200: "#FECDD3",
              300: "#FDA4AF",
              400: "#FB7185",
              500: "#E50914",
              600: "#E11D48",
              700: "#BE123C",
              800: "#9F1239",
              900: "#881337",
              DEFAULT: "#E50914",
              foreground: "#FFFFFF",
            },
            focus: "#E50914",
          },
        },
      },
    }),
  ],
};

export default config;
