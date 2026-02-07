# ViFactCheck Client

A modern, premium React-based web application for verifying Vietnamese news and claims using the ViFactCheck API.

## Overview

ViFactCheck Client is a sleek, responsive fact-checking interface that integrates with the ViFactCheck API backend. It provides an intuitive chat-like interface where users can submit claims or statements to be verified against a database of evidence.

## Features

- **Real-time Fact Checking**: Submit claims and get instant verification with detailed explanations
- **Comprehensive Results Display**:
  - Truth status (True/False/Unverified) with color-coded badges
  - Confidence score with visual progress bar
  - Detailed explanation from the AI fact-checker
  - Evidence sources with trust levels and relevance scores
- **Premium Design**:
  - Modern gradient design with indigo color scheme
  - Smooth animations and transitions
  - Responsive layout optimized for all screen sizes
  - Custom scrollbar styling
  - Professional typography using Google Fonts (Inter)
- **User-Friendly Interface**:
  - Chat-like interaction pattern
  - Loading indicators during API calls
  - Error handling with user-friendly messages
  - Disabled state during processing

## Tech Stack

- **React 19**: Modern UI library
- **Vite**: Fast build tool and dev server (Rolldown-powered)
- **TailwindCSS**: Utility-first CSS framework
- **Google Fonts**: Inter font family for premium typography

## Getting Started

### Prerequisites

- Node.js 16+ installed
- ViFactCheck API server running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or another port if 5173 is in use).

### Build for Production

```bash
npm run build
```

The production-ready files will be in the `dist` directory.

## API Configuration

The client is configured to connect to the ViFactCheck API at `http://localhost:8000`. 

To change the API endpoint, edit `src/services/chatService.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000'  // Change this to your API URL
```

## API Integration

The application integrates with the following ViFactCheck API endpoints:

### 1. Health Check
- **Endpoint**: `GET /`
- **Purpose**: Verify API server is running

### 2. Verify Claim
- **Endpoint**: `POST /check`
- **Request Body**:
  ```json
  {
    "claim": "Your claim to verify"
  }
  ```
- **Response**: Returns verification result with:
  - `status`: Truth verdict (True/False/Unverified)
  - `explanation`: Detailed explanation from the LLM
  - `confidence`: Confidence score (0.0 - 1.0)
  - `evidence`: Array of evidence objects with sources, scores, and trust levels

For complete API documentation, see `docs/API_DOCUMENTATION.md`.

## Project Structure

```
client/
├── src/
│   ├── components/
│   │   ├── ChatInput.jsx      # Input field with submit button
│   │   ├── Message.jsx         # Message display (user + fact-check results)
│   │   └── NavBar.jsx          # Navigation bar with branding
│   ├── services/
│   │   └── chatService.js      # API integration layer
│   ├── App.jsx                 # Main application component
│   ├── index.css               # Global styles and Tailwind imports
│   └── main.jsx               # Application entry point
├── docs/
│   └── API_DOCUMENTATION.md    # Complete API documentation
├── public/                     # Static assets
├── index.html                  # HTML template
└── package.json               # Project dependencies
```

## Usage

1. **Start a Verification**: Type a claim or statement in Vietnamese into the input field
2. **Submit**: Click "Kiểm tra" (Check) or press Enter
3. **Review Results**: The system will display:
   - The verification status (True/False/Unverified)
   - Confidence level
   - Detailed explanation
   - Supporting evidence with sources

### Example Claims to Test

```
Vụ cháy chung cư mini Khương Hạ nguyên nhân do chập điện xe máy.
```

## Design System

### Colors
- **Primary**: Indigo (600-800) for main interface elements
- **Success**: Green for "True" verdicts
- **Danger**: Red for "False" verdicts
- **Warning**: Yellow for "Unverified" status
- **Neutral**: Gray shades for text and backgrounds

### Typography
- **Font Family**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700, 800

### Components
All components follow a consistent design pattern with:
- Rounded corners (rounded-2xl, rounded-xl)
- Shadow effects for depth
- Hover states with smooth transitions
- Gradient backgrounds for premium feel

## Error Handling

The application handles various error scenarios:

1. **Network Errors**: Displays user-friendly message if API is unreachable
2. **API Errors**: Shows error details from the backend
3. **Invalid Input**: Prevents submission of empty claims
4. **Loading States**: Disables input during processing

## Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

When making changes to the UI:

1. Maintain the existing design language and color scheme
2. Ensure all text displays properly in Vietnamese
3. Test with the actual API to verify data handling
4. Follow React best practices and functional component patterns

## Notes

- The `@tailwind` warnings in the CSS file are expected and can be ignored - they're standard TailwindCSS directives
- The application uses modern ES6+ features - ensure your Node.js version supports them
- Custom scrollbar styling may not work in all browsers (falls back to default)

## License

This project is part of the Text Mining Application Final Project.
