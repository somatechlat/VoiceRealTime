# AgentVoiceBox Portal Design Guidelines

## Design References

Based on Fish Audio, Twisty Dashboard, Verve Dashboard, and modern SaaS UI patterns.

### Light Theme Reference: Verve Dashboard
- Soft sage-tinted white background
- Pill-shaped navigation with filled active states
- Large rounded cards (16-20px radius)
- Muted teal/mint accent colors for highlights
- Large bold metric numbers with small muted labels
- Soft gradient area charts
- Avatar-based user profiles in header
- Message/activity lists with notification badges

## Core Principles

1. **Simple** - No visual clutter, only essential elements
2. **Elegant** - Refined typography, subtle shadows, smooth transitions
3. **Human-Friendly** - Clear icons, readable text, intuitive flows
4. **Minimal** - Generous whitespace, focused content areas

---

## Color Palette

### Dark Theme
```css
--background: #0a0a0f        /* Deep black */
--card: #111118              /* Slightly lighter card */
--card-hover: #16161d        /* Card hover state */
--border: #1e1e26            /* Subtle borders */
--text-primary: #e4e4e7      /* Primary text */
--text-secondary: #71717a    /* Secondary/muted text */
--primary: #3b82f6           /* Blue accent */
--primary-hover: #2563eb     /* Blue hover */
--success: #22c55e           /* Green for success */
--warning: #f59e0b           /* Amber for warnings */
--error: #ef4444             /* Red for errors */
```

### Light Theme
```css
--background: #f5f7f5        /* Soft sage-tinted white (Verve style) */
--background-alt: #f8fafc    /* Alternative soft gray-blue */
--card: #ffffff              /* White cards */
--card-hover: #f1f5f9        /* Card hover state */
--card-accent: #e8f5f0       /* Soft teal/mint card background */
--border: #e2e8f0            /* Light borders */
--text-primary: #0f172a      /* Dark text */
--text-secondary: #64748b    /* Secondary/muted text */
--text-muted: #94a3b8        /* Very muted text for labels */
--primary: #2563eb           /* Blue accent */
--primary-hover: #1d4ed8     /* Blue hover */
--accent-teal: #0d9488       /* Teal accent (Verve style) */
--accent-teal-light: #ccfbf1 /* Light teal for backgrounds */
--success: #16a34a           /* Green for success */
--warning: #d97706           /* Amber for warnings */
--error: #dc2626             /* Red for errors */
```

---

## Typography

### Font Family
- Primary: Inter (system fallback: -apple-system, BlinkMacSystemFont, sans-serif)

### Font Sizes
```css
--text-xs: 12px      /* Small labels, badges */
--text-sm: 14px      /* Secondary text, table cells */
--text-base: 16px    /* Body text */
--text-lg: 18px      /* Subheadings */
--text-xl: 20px      /* Section titles */
--text-2xl: 24px     /* Page titles */
--text-3xl: 30px     /* Hero numbers */
```

### Font Weights
- Regular: 400 (body text)
- Medium: 500 (labels, buttons)
- Semibold: 600 (headings)
- Bold: 700 (emphasis, large numbers)

---

## Spacing Scale

```css
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-5: 20px
--space-6: 24px
--space-8: 32px
--space-10: 40px
--space-12: 48px
--space-16: 64px
```

---

## Border Radius

```css
--radius-sm: 4px     /* Small elements, badges */
--radius-md: 8px     /* Buttons, inputs */
--radius-lg: 12px    /* Cards (dark theme) */
--radius-xl: 16px    /* Large cards, modals */
--radius-2xl: 20px   /* Extra large cards (Verve light style) */
--radius-full: 9999px /* Pills, avatars, nav items */
```

---

## Shadows

### Dark Theme
```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3)
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.4)
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5)
```

### Light Theme
```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05)
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07)
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1)
```

---

## Components

### Cards
- Dark theme: 12px radius, subtle border
- Light theme: 16-20px radius (Verve style), soft shadow
- Padding: 24px
- Hover: slight background change
- Accent cards: Use `--card-accent` for highlighted metrics

### Metric Cards (Verve Style)
- Large bold number (30px, font-weight 700)
- Small muted label above (12px, `--text-muted`)
- Optional percentage badge with trend indicator
- Optional soft teal/mint background for emphasis
- Progress bar below metric (optional)

### Buttons

#### Primary Button
- Background: primary color
- Text: white
- Padding: 12px 24px
- Border radius: 8px
- Hover: darker shade
- Transition: 150ms ease

#### Secondary Button
- Background: transparent
- Border: 1px solid border color
- Text: primary text color
- Hover: subtle background

#### Ghost Button
- Background: transparent
- No border
- Text: secondary text color
- Hover: subtle background

### Inputs
- Height: 44px
- Border radius: 8px
- Border: 1px solid border color
- Focus: primary color ring
- Padding: 12px 16px

### Theme Toggle
- Icon-based (Sun/Moon)
- Smooth transition between states
- Position: Header right side
- Size: 40px touch target

---

## Layout

### Navigation (Verve Style - Light Theme)
- Fixed header: 64px height
- Logo left, pill-shaped nav center, user profile right
- Nav items: pill-shaped with `border-radius: 9999px`
- Active nav: filled background with icon + text
- Inactive nav: text only, hover shows subtle background
- User profile: avatar + name + email dropdown

### Navigation (Dark Theme)
- Fixed header: 64px height
- Logo left, nav center, actions right
- Clean horizontal links with underline active state
- User avatar with dropdown

### Sidebar (Admin Portal)
- Width: 240px collapsed, 280px expanded
- Icon + text navigation items
- Collapsible on mobile

### Content Area
- Max width: 1280px
- Padding: 24px (desktop), 16px (mobile)
- Card-based sections

### Grid
- 12-column grid
- Gap: 24px
- Responsive breakpoints:
  - Mobile: < 640px
  - Tablet: 640px - 1024px
  - Desktop: > 1024px

---

## Animations

### Transitions
```css
--transition-fast: 100ms ease
--transition-base: 150ms ease
--transition-slow: 300ms ease
```

### Hover Effects
- Cards: subtle background shift
- Buttons: color change
- Links: underline or color change

### Loading States
- Skeleton screens matching content shape
- Subtle pulse animation
- No jarring spinners

---

## Icons

### Library
- Lucide React icons
- Consistent 20px size
- 1.5px stroke width

### Common Icons
- Dashboard: LayoutDashboard
- API Keys: Key
- Billing: CreditCard
- Team: Users
- Settings: Settings
- Logout: LogOut
- Sun: Sun (light mode)
- Moon: Moon (dark mode)
- Search: Search
- Menu: Menu
- Close: X
- Check: Check
- Error: AlertCircle
- Info: Info

---

## Login Page Specifications

### Layout
- Centered card on gradient background
- Logo at top
- Form fields stacked vertically
- Social login options below
- Footer links at bottom

### Elements
1. Logo (AgentVoiceBox)
2. Welcome heading
3. Subtext
4. Email input
5. Password input with show/hide toggle
6. "Remember me" checkbox
7. "Forgot password?" link
8. Primary login button
9. Divider with "or"
10. Social login buttons (Google, GitHub)
11. "Don't have an account? Sign up" link

### Background
- Dark: Subtle gradient from #0a0a0f to #111118
- Light: Soft gradient from #f5f7f5 to #e8f5f0 (sage-tinted)

---

## Dashboard Components (Verve Style)

### Stats Row
- 2-4 metric cards in a row
- Each shows: label, large number, percentage change
- Some cards use accent background (`--card-accent`)

### Charts
- Area charts: soft gradient fill (teal to transparent)
- Bar charts: rounded tops, subtle colors
- Line charts: smooth curves, dot markers on hover

### Activity/Messages List
- Avatar (40px, rounded full)
- Name + preview text
- Notification badge (colored circle with count)
- Timestamp or status indicator

### Search Bar
- Rounded pill shape (`border-radius: 9999px`)
- Search icon left, filter icon right
- Placeholder: "Search..." in muted text
- Light background in light theme

### Date Display
- Format: "Monday, 12 August, 2024"
- Position: Top left of dashboard
- Font: 14px, `--text-secondary`

---

## Accessibility

### Focus States
- Visible focus ring (2px primary color)
- Skip links for keyboard navigation
- Proper tab order

### Color Contrast
- WCAG 2.1 AA minimum (4.5:1 for text)
- 3:1 for UI components

### Screen Readers
- Proper ARIA labels
- Semantic HTML
- Alt text for images

---

## Responsive Breakpoints

```css
/* Mobile first */
@media (min-width: 640px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 1024px) { /* lg */ }
@media (min-width: 1280px) { /* xl */ }
```
