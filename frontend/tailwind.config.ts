import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        evergreen: '#042a2b',
        'light-cyan': '#cdedf6',
        'pacific-blue': '#5eb1bf',
        'atomic-tangerine': '#ef7b45',
        'burnt-tangerine': '#d84727',
      },
      fontFamily: {
        display: ['"Open Sans"', 'sans-serif'],
        sans: ['"Open Sans"', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '0.5rem',
        lg: '0.75rem',
        xl: '1rem',
      },
    },
  },
}
export default config
