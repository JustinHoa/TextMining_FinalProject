# UI Component Guide

This document describes the UI components configured for the ViFactCheck application based on the API documentation.

## Visual Design Language

### Color Palette
- **Primary Brand Color**: Indigo (#4F46E5 - #6366F1)
- **Status Colors**:
  - True/Success: Green (#10B981)
  - False/Error: Red (#EF4444)
  - Unverified/Warning: Yellow (#EAB308)
- **Neutral Colors**: Gray scale for backgrounds and text

### Typography
- **Font**: Inter (Google Fonts)
- **Heading Sizes**: 2xl for main titles, sm-lg for content
- **Weights**: Regular (400) for body, Medium (500-600) for labels, Bold (700) for headings

## Component Breakdown

### 1. NavBar Component
**Location**: Top of the page  
**Features**:
- ViFactCheck branding with shield icon
- Gradient background (indigo 600-800)
- API status indicator (green pulsing dot)
- Tagline: "Kiểm tra thông tin tự động"

**Design Elements**:
- Rounded corners (rounded-2xl)
- Shadow for depth
- Glassmorphism effect on logo and status indicator
- Responsive layout

### 2. Message Component (User Input)
**Location**: Chat area  
**Features**:
- Right-aligned for user messages
- Gradient background (indigo 600-700)
- White text
- Compact size with shadow

**Design Elements**:
- Rounded pill shape (rounded-3xl)
- Maximum width constraint
- Subtle shadow for depth

### 3. Message Component (Fact-Check Result)
**Location**: Chat area  
**Features**:
This is the most complex component, displaying structured fact-check results.

#### Header Section
- **Status Badge**: Color-coded pill (Green/Red/Yellow) showing True/False/Unverified
- **Confidence Bar**: Visual progress bar with percentage
  - Green: ≥80%
  - Yellow: 50-79%
  - Red: <50%
- **Claim Display**: White box with indigo left border showing the original claim

#### Explanation Section
- **Icon**: Document icon in indigo
- **Title**: "Giải thích:" (Explanation)
- **Content**: AI-generated explanation text
- **Background**: Light gray for separation

#### Evidence Section
- **Title**: "Bằng chứng (X):" showing count
- **Evidence Cards**: For each piece of evidence:
  - Trust level badge (High/Medium/Unknown)
  - Source and relevance score
  - Statement/headline
  - Evidence snippet (truncated to 3 lines)
  - Link to source (if available)

**Design Elements**:
- White background with shadow
- Rounded corners (rounded-2xl for container, rounded-xl for cards)
- Hover effects on evidence cards
- Color-coded trust levels
- Responsive layout
- Full-width display (max-w-3xl)

### 4. ChatInput Component
**Location**: Bottom of the page  
**Features**:
- Text input field
- Submit button with Vietnamese text "Kiểm tra"
- Disabled state during API calls
- Placeholder in Vietnamese

**Design Elements**:
- White container with shadow
- Rounded corners (rounded-2xl)
- Gradient button (indigo 600-700)
- Focus states with ring effect
- Disabled state with reduced opacity
- Active scale animation on button

### 5. Loading Indicator
**Location**: Chat area (appears during API calls)  
**Features**:
- Three animated dots in indigo
- Vietnamese text: "Đang kiểm tra thông tin..."
- White background with border

**Design Elements**:
- Bounce animation with staggered delays
- Rounded container
- Minimalist design

## Layout Structure

```
┌─────────────────────────────────────────────────────┐
│                      NavBar                         │
│  ViFactCheck | Kiểm tra thông tin tự động  [Ready] │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   Message Area                      │
│                                                     │
│  ┌─────────────────────────────────────────┐       │
│  │ Assistant: Welcome message              │       │
│  └─────────────────────────────────────────┘       │
│                                                     │
│       ┌─────────────────────────────────────┐      │
│       │ User: Claim to verify               │      │
│       └─────────────────────────────────────┘      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ [True] 95% ┌──────────────────────────┐    │   │
│  │            │ Claim: ...               │    │   │
│  │            └──────────────────────────┘    │   │
│  │                                             │   │
│  │ Giải thích:                                 │   │
│  │ Explanation text...                         │   │
│  │                                             │   │
│  │ Bằng chứng (2):                             │   │
│  │ ┌─────────────────────────────────────┐    │   │
│  │ │ [High] Local-DB • 88%               │    │   │
│  │ │ Statement...                        │    │   │
│  │ │ Evidence chunk...                   │    │   │
│  │ │ [Xem nguồn]                         │    │   │
│  │ └─────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ┌────────────────────────────┐ ┌──────────────────┐│
│ │ Nhập tuyên bố...           │ │  Kiểm tra       ││
│ └────────────────────────────┘ └──────────────────┘│
└─────────────────────────────────────────────────────┘
```

## Responsive Behavior

### Desktop (>1260px)
- Maximum width: 1260px
- Full component display
- Evidence cards in full detail

### Tablet (768px - 1260px)
- Flexible width with padding
- Adjusted font sizes
- Evidence cards maintain layout

### Mobile (<768px)
- Full-width layout
- Stacked evidence cards
- Adjusted padding and spacing
- Touch-friendly button sizes

## Interactive States

### Hover States
- Evidence cards: Border color changes to indigo-300, shadow increases
- Buttons: Background darkens
- Links: Color darkens, underline appears

### Focus States
- Input fields: Indigo ring with border highlight
- Buttons: Ring with offset for visibility

### Disabled States
- Input and button: 50% opacity
- Cursor: not-allowed
- Scale animation disabled

### Loading States
- Input disabled
- Animated dots indicator
- Submit button disabled

## Accessibility Features

1. **Semantic HTML**: Proper use of nav, form, button elements
2. **ARIA Labels**: Appropriate for screen readers
3. **Focus Management**: Clear focus states with rings
4. **Color Contrast**: WCAG AA compliant
5. **Keyboard Navigation**: Full keyboard support

## Animation Timing

- **Button Scale**: 0.95 on active
- **Hover Transitions**: 0.15s ease
- **Loading Dots**: Bounce with 150ms stagger
- **Status Indicator**: Pulse animation (continuous)

## Integration with API

The UI is specifically designed to map to the API response structure:

```javascript
// API Response Structure
{
  claim: string,
  status: "True" | "False" | "Unverified",
  explanation: string,
  confidence: number (0.0-1.0),
  evidence: [{
    source: string,
    score: number,
    evidence_chunk: string,
    context_summary: string,
    statement: string,
    url: string,
    trust_level: "High" | "Medium" | "Unknown"
  }]
}
```

Each field from the API is displayed in a dedicated section of the Message component, ensuring all information is presented clearly and hierarchically.

## Best Practices

1. **Always show loading states** during API calls
2. **Display full error messages** from the API
3. **Truncate long evidence** with expand option (currently 3 lines)
4. **Maintain consistent spacing** using Tailwind utilities
5. **Use semantic colors** for status (green=true, red=false, yellow=unverified)
6. **Provide visual hierarchy** with font weights and sizes
7. **Ensure responsive design** for all screen sizes

## Future Enhancements

Potential improvements for the UI:
- Expand/collapse for long evidence chunks
- Copy-to-clipboard for claims and explanations
- Share functionality for results
- Dark mode toggle
- Evidence source filtering
- History of past verifications
- Export results as PDF/image
