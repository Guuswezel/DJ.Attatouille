/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        glow: '0 0 0 1px rgba(255,255,255,.08), 0 20px 50px rgba(0,0,0,.32)',
      },
      colors: {
        ink: '#0a0a0b',
        panel: '#151518',
        lime: '#d9ff4d',
        coral: '#ff705f',
      },
    },
  },
  plugins: [],
}

