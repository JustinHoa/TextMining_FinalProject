# ViFactCheck Client

Frontend application for the ViFactCheck fact-checking system. Built with React, Vite, and TailwindCSS.

## 🚀 Features

- **Modern UI**: Clean, responsive interface built with React
- **Real-time Fact Checking**: Submit claims and get instant verification
- **Visual Feedback**: Clear display of verification status and evidence
- **Tailwind Styling**: Beautiful, utility-first CSS framework

## 📋 Prerequisites

- **Node.js**: Version 16 or higher
- **npm**: Version 7 or higher (comes with Node.js)

Check your versions:
```bash
node --version
npm --version
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

From the client directory, install all required packages:

```bash
npm install
```

This will install:
- React and React DOM
- Vite (build tool)
- TailwindCSS (styling)
- ESLint (code linting)
- All other dependencies defined in `package.json`

### 2. Configure API Endpoint (Optional)

The client is pre-configured to connect to the backend at `http://localhost:8000`.

If your backend runs on a different port, update the API URL in your service file:
- Edit `src/services/api.js` (or wherever API calls are defined)
- Change the base URL to match your backend

## 🏃 Running the Application

### Development Mode

Start the development server with hot module replacement:

```bash
npm run dev
```

The application will start at: `http://localhost:5173` (Vite's default port)

The development server features:
- ⚡ Lightning-fast hot module replacement (HMR)
- 🔄 Auto-reload on file changes
- 📊 Detailed error messages

### Build for Production

Create an optimized production build:

```bash
npm run build
```

This generates a `dist/` folder with optimized static files.

### Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

### Linting

Run ESLint to check code quality:

```bash
npm run lint
```

## 🌐 Usage

### Basic Workflow

1. **Start the Backend Server** (see server README)
   ```bash
   # In the server directory
   uvicorn main:app --reload
   ```

2. **Start the Frontend** (in a new terminal)
   ```bash
   # In the client directory
   npm run dev
   ```

3. **Open Browser**: Navigate to `http://localhost:5173`

4. **Enter a Claim**: Type your claim in the input field

5. **Get Results**: View the verification status, confidence score, and supporting evidence

### Example Claims to Test

- "Việt Nam là thành viên của ASEAN"
- "COVID-19 xuất hiện lần đầu tiên năm 2019"
- "Trái đất phẳng"

## 📁 Project Structure

```
client/
├── index.html              # HTML entry point
├── package.json            # Project configuration & dependencies
├── vite.config.js          # Vite configuration
├── tailwind.config.js      # TailwindCSS configuration
├── postcss.config.js       # PostCSS configuration
├── eslint.config.js        # ESLint configuration
├── public/                 # Static assets
└── src/                    # Source code
    ├── main.jsx            # Application entry point
    ├── App.jsx             # Main App component
    ├── index.css           # Global styles
    ├── components/         # React components
    │   ├── ClaimInput.jsx
    │   ├── ResultDisplay.jsx
    │   └── EvidenceCard.jsx
    └── services/           # API services
        └── api.js          # API communication
```

## ⚙️ Configuration

### Vite Configuration

The `vite.config.js` includes React plugin configuration. You can modify:
- Port number
- Proxy settings for API calls
- Build optimizations

Example custom port:
```javascript
export default defineConfig({
  server: {
    port: 3000
  }
})
```

### TailwindCSS

Tailwind is configured in `tailwind.config.js`. You can customize:
- Color palette
- Fonts
- Breakpoints
- Custom utilities

### Environment Variables

Create a `.env` file in the client directory for environment-specific settings:

```env
VITE_API_URL=http://localhost:8000
```

Access in code:
```javascript
const apiUrl = import.meta.env.VITE_API_URL;
```

## 🎨 Styling

The project uses **TailwindCSS** for styling. Key concepts:

### Utility Classes
```jsx
<div className="flex items-center justify-center p-4 bg-blue-500">
  <button className="px-6 py-2 text-white rounded-lg hover:bg-blue-600">
    Check Claim
  </button>
</div>
```

### Custom Styles
Add custom CSS in `src/index.css`:
```css
@layer components {
  .btn-primary {
    @apply px-6 py-2 bg-blue-500 text-white rounded-lg;
  }
}
```

## 🔧 Troubleshooting

### Port Already in Use

If port 5173 is in use:
```bash
# Use a different port
npm run dev -- --port 3000
```

### Module Not Found Errors

Clear cache and reinstall:
```bash
# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall dependencies
npm install
```

### Build Errors

Clear Vite cache:
```bash
rm -rf node_modules/.vite
npm run dev
```

### CORS Errors

Ensure the backend server has CORS properly configured for your frontend URL:
```python
# In server/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API Connection Issues

1. **Verify backend is running**: Check `http://localhost:8000` in browser
2. **Check API URL**: Ensure it matches your backend port
3. **Browser console**: Look for error messages (F12 → Console)
4. **Network tab**: Check if API calls are being made (F12 → Network)

## 📱 Responsive Design

The application is responsive and works on:
- 📱 Mobile devices (320px+)
- 📱 Tablets (768px+)
- 💻 Desktops (1024px+)
- 🖥️ Large screens (1440px+)

Test responsiveness:
- Browser DevTools (F12 → Responsive Design Mode)
- Different browser window sizes
- Actual devices

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Deploy to Static Hosting

The `dist/` folder can be deployed to:
- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop the `dist/` folder
- **GitHub Pages**: Use `gh-pages` package
- **AWS S3**: Upload to S3 bucket

### Example: Vercel Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

## 🧪 Testing

### Manual Testing Checklist

- [ ] Submit a valid claim
- [ ] Submit an empty claim (should show error)
- [ ] Test with Vietnamese text
- [ ] Test with very long claims
- [ ] Check loading states
- [ ] Verify error messages
- [ ] Test on different browsers
- [ ] Test on mobile devices

## 🔌 API Integration

The client communicates with the backend via REST API:

### Check Endpoint

```javascript
// POST /check
const response = await fetch('http://localhost:8000/check', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    claim: 'Your claim here'
  })
});

const data = await response.json();
// Returns: { claim, status, explanation, confidence, evidence }
```

## 🤝 Development Tips

### Hot Reload
Vite provides instant hot module replacement. Changes appear immediately in the browser.

### React DevTools
Install React DevTools browser extension for debugging:
- Chrome: [React DevTools](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- Firefox: [React DevTools](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### Code Formatting
Use Prettier for consistent formatting:
```bash
npm install --save-dev prettier
npx prettier --write src/
```

## 📚 Learning Resources

- **React**: https://react.dev
- **Vite**: https://vitejs.dev
- **TailwindCSS**: https://tailwindcss.com
- **MDN Web Docs**: https://developer.mozilla.org

## 🤝 Contributing

When making changes:
1. Follow React best practices
2. Use functional components and hooks
3. Keep components small and focused
4. Add PropTypes or TypeScript for type checking
5. Test on multiple browsers
6. Update this README if needed

## 📄 License

This project is part of a Text Mining Application assignment.
