# ViFactCheck UI Configuration Summary

This document summarizes the UI configuration changes made to align with the ViFactCheck API documentation.

## 🎯 Overview

The UI has been completely configured to integrate with the ViFactCheck API as specified in `docs/API_DOCUMENTATION.md`. The application now provides a premium, modern interface for fact-checking Vietnamese news and claims.

## 📋 Changes Made

### 1. **API Integration** (`src/services/chatService.js`)
- ✅ Implemented `verifyClaimAPI()` function to call `/check` endpoint
- ✅ Implemented `checkAPIHealth()` function to call `/` endpoint
- ✅ Proper error handling and response parsing
- ✅ Removed mock responses

### 2. **Message Component** (`src/components/Message.jsx`)
- ✅ Enhanced to display structured fact-check results
- ✅ Shows status badge with color coding (True=Green, False=Red, Unverified=Yellow)
- ✅ Displays confidence score with visual progress bar
- ✅ Shows detailed explanation from the LLM
- ✅ Renders evidence with:
  - Trust level badges
  - Relevance scores
  - Evidence chunks with truncation
  - Clickable source links
- ✅ Maintains simple message display for non-fact-check messages

### 3. **App Component** (`src/App.jsx`)
- ✅ Integrated with `verifyClaimAPI()` service
- ✅ Added loading state during API calls
- ✅ Implemented error handling with user-friendly messages
- ✅ Added animated loading indicator
- ✅ Updated welcome message to Vietnamese
- ✅ Changed background to modern gradient

### 4. **ChatInput Component** (`src/components/ChatInput.jsx`)
- ✅ Added `disabled` prop support
- ✅ Updated placeholder to Vietnamese
- ✅ Changed button text to "Kiểm tra" (Check)
- ✅ Improved styling with indigo gradient
- ✅ Added disabled state styling

### 5. **NavBar Component** (`src/components/NavBar.jsx`)
- ✅ Updated branding to "ViFactCheck"
- ✅ Changed shield icon to match fact-checking theme
- ✅ Added tagline "Kiểm tra thông tin tự động"
- ✅ Added API status indicator
- ✅ Updated color scheme to indigo gradient

### 6. **Styling** (`src/index.css`)
- ✅ Added Google Fonts (Inter) for premium typography
- ✅ Updated scrollbar styling to indigo theme
- ✅ Added line-clamp utility for text truncation
- ✅ Improved overall design consistency

### 7. **HTML Template** (`index.html`)
- ✅ Updated title to "ViFactCheck - Kiểm tra thông tin tự động"
- ✅ Changed language to Vietnamese (`lang="vi"`)
- ✅ Added meta description
- ✅ Added SEO keywords

### 8. **Documentation**
- ✅ Created comprehensive `README.md` with setup instructions
- ✅ Created `docs/UI_COMPONENT_GUIDE.md` with design system documentation

## 🎨 Design Theme

### Color Palette
- **Primary**: Indigo (#4F46E5 - #6366F1)
- **Success**: Green (#10B981) - for True verdicts
- **Danger**: Red (#EF4444) - for False verdicts  
- **Warning**: Yellow (#EAB308) - for Unverified status
- **Backgrounds**: White and gray gradients

### Typography
- **Font**: Inter (Google Fonts)
- **Sizes**: Responsive scaling from xs to 2xl
- **Weights**: 300, 400, 500, 600, 700, 800

### Visual Elements
- Rounded corners (rounded-2xl, rounded-xl)
- Gradient backgrounds
- Shadow effects for depth
- Smooth transitions and animations
- Hover effects on interactive elements

## 🔌 API Mapping

The UI components directly map to the API response structure:

```
API Response                    →    UI Component
────────────────────────────────────────────────────
claim                          →    Claim display box
status                         →    Color-coded badge
explanation                    →    Explanation section
confidence                     →    Progress bar + percentage
evidence[]                     →    Evidence cards
  ├─ source                    →    Source label
  ├─ score                     →    Relevance percentage
  ├─ evidence_chunk            →    Truncated text
  ├─ statement                 →    Card title
  ├─ url                       →    "Xem nguồn" link
  └─ trust_level               →    Trust badge
```

## 🚀 Quick Start

1. **Ensure API is running**:
   ```bash
   # Backend should be running on http://localhost:8000
   ```

2. **Start the client**:
   ```bash
   npm run dev
   ```

3. **Test the integration**:
   - Open http://localhost:5173
   - Enter a Vietnamese claim
   - Click "Kiểm tra"
   - Review the fact-check results

## 🧪 Testing the UI

### Test Cases

1. **Valid Claim**:
   - Input: `Vụ cháy chung cư mini Khương Hạ nguyên nhân do chập điện xe máy.`
   - Expected: Structured result with True/False status, explanation, and evidence

2. **API Unavailable**:
   - Ensure API is stopped
   - Input any claim
   - Expected: Error message displayed

3. **Empty Input**:
   - Try submitting without text
   - Expected: Button disabled, no submission

4. **Loading State**:
   - Submit a claim
   - Expected: Loading indicator appears, input disabled

## 📊 Component Hierarchy

```
App
├── NavBar
│   ├── Logo (Shield icon)
│   ├── Brand name
│   └── API status indicator
├── Messages Container
│   ├── Message (Assistant - welcome)
│   ├── Message (User - claim)
│   ├── Message (Assistant - fact-check result)
│   │   ├── Status badge
│   │   ├── Confidence bar
│   │   ├── Claim display
│   │   ├── Explanation section
│   │   └── Evidence section
│   │       └── Evidence cards (multiple)
│   └── Loading Indicator (conditional)
└── ChatInput
    ├── Input field
    └── Submit button
```

## ✨ Key Features

1. **Real-time fact checking** via API integration
2. **Comprehensive result display** with all API fields
3. **Loading states** with animated indicators
4. **Error handling** with Vietnamese messages
5. **Responsive design** for all screen sizes
6. **Premium aesthetics** with modern gradients and typography
7. **Accessibility** with proper focus states and semantic HTML
8. **SEO optimization** with meta tags and descriptions

## 🔧 Configuration

### To Change API URL:
Edit `src/services/chatService.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000'  // Change this
```

### To Modify Colors:
Edit component className props with TailwindCSS utilities, or update `src/index.css` for global changes.

### To Adjust Confidence Thresholds:
Edit `src/components/Message.jsx`, look for:
```javascript
confidence >= 0.8 ? 'bg-green-500' : 
confidence >= 0.5 ? 'bg-yellow-500' : 
'bg-red-500'
```

## 📝 Notes

- The UI is fully Vietnamese localized for user-facing text
- All API response fields are displayed
- Evidence chunks are truncated to 3 lines for better UX
- The design follows modern web design best practices
- Animations are subtle and performance-optimized

## 🐛 Known Issues / Limitations

1. **@tailwind lint warnings**: These are expected and can be safely ignored - they're standard TailwindCSS directives
2. **Custom scrollbar**: May not display in all browsers (falls back to default)
3. **Evidence expansion**: Currently no expand/collapse for long evidence (could be future enhancement)

## 📚 Related Documentation

- `docs/API_DOCUMENTATION.md` - Complete API reference
- `docs/UI_COMPONENT_GUIDE.md` - Detailed UI component guide
- `README.md` - Project setup and usage

## 🎉 Result

The UI is now fully configured and aligned with the ViFactCheck API documentation. It provides a modern, premium interface for Vietnamese fact-checking with comprehensive display of all verification results.
