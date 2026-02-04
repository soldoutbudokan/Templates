import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ["Georgia", "Cambria", "Times New Roman", "Times", "serif"],
      },
      colors: {
        orwell: {
          cream: "#FAF8F5",
          paper: "#F5F2ED",
          ink: "#2C2C2C",
          accent: "#8B4513",
          muted: "#6B6B6B",
        },
      },
    },
  },
  plugins: [],
};
export default config;
