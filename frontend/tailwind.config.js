/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                display: ['Space Grotesk', 'sans-serif'],
            },
            colors: {
                primary: {
                    500: '#3b82f6', // Blue
                    600: '#2563eb',
                },
                accent: {
                    500: '#8b5cf6', // Purple
                    600: '#7c3aed',
                }
            },
            animation: {
                'scan': 'scan 2s linear infinite',
            },
            keyframes: {
                scan: {
                    '0%': { top: '0%' },
                    '100%': { top: '100%' },
                }
            }
        },
    },
    plugins: [],
}
