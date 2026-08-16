# Template Information: Activity 01 - Introduction to Web Design

## 📁 File Structure

```
activity-01-web-design-basics/
├── index.html              # Main HTML file with TODOs
├── style.css               # CSS stylesheet with TODOs
├── README.md               # Student-facing instructions
├── TEACHER_INSTRUCTIONS.mdx # Pedagogical guide
└── TEMPLATE-INFO.md        # This file
```

## 🎯 Template Purpose

This template introduces students to fundamental web design concepts through a partially completed webpage. It demonstrates:
- Basic HTML document structure
- Navigation between sections using anchor links
- CSS styling with classes and IDs
- Responsive design principles
- Modern CSS features (Flexbox, Grid, gradients)

## 🏗️ Architecture Breakdown

### HTML Structure

**Complete Sections (65-70%):**
1. **Document Head**
   - Meta tags for charset and viewport
   - Title and CSS link
   - Fully functional, no student changes needed

2. **Header Section**
   - `<header>` with title and subtitle
   - Pre-styled with gradient theme
   - Demonstrates semantic HTML

3. **Navigation Menu**
   - `<nav>` with unordered list
   - Working anchor links to sections
   - Hover effects pre-implemented

4. **About Section**
   - Introduction to web design
   - Pre-styled with `.section` class
   - **TODO:** Add website vs webpage paragraph

5. **Key Concepts Section**
   - Grid layout for concept cards
   - Two cards pre-built (HTML, CSS)
   - **TODO:** Add third JavaScript card

6. **Footer**
   - Copyright information
   - Fully styled and complete

**Incomplete Sections (30-35%):**
1. **Practice Section**
   - **TODO:** Create entire section
   - Should include heading, list, and link

2. **Explanatory Content**
   - **TODO:** Add website vs webpage explanation

### CSS Structure

**Complete Styles (65-70%):**
1. **Global Styles**
   - Reset with universal selector
   - Body styling with gradient background
   - Typography settings

2. **Header Styles**
   - White background with transparency
   - Purple color scheme
   - Box shadow for depth

3. **Navigation Styles**
   - Flexbox for horizontal layout
   - Hover effects with color transition
   - Responsive design

4. **Grid Layout**
   - Responsive grid for concept cards
   - Auto-fit with minmax for flexibility

5. **Typography**
   - Heading styles
   - Color hierarchy

6. **Footer Styles**
   - Centered text
   - Matching color scheme

7. **Responsive Breakpoints**
   - Mobile-first approach
   - 768px breakpoint for tablets/mobile

**Incomplete Styles (30-35%):**
1. **Section Class Styles**
   - **TODO:** Add background, spacing, shadows

2. **Concept Card Styles**
   - **TODO:** Add gradient background, padding, shadows
   - **TODO:** Add hover transform effect

## 🎨 Design System

### Color Palette
- **Primary Purple:** `#667eea` - Used for headings, navigation
- **Secondary Purple:** `#764ba2` - Gradient end color
- **Background Gradient:** `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
- **White Transparency:** `rgba(255, 255, 255, 0.95)` - Sections
- **Text Color:** `#333` - Body text
- **Light Gray:** `#666` - Secondary text (subtitle, footer)

### Typography
- **Font Family:** 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
- **Line Height:** 1.6 for readability
- **Heading Sizes:**
  - H1: 2.5rem (2rem on mobile)
  - H2: 1.8rem
  - H3: 1.3rem

### Spacing
- **Section Margin:** 2rem auto
- **Section Padding:** 2rem
- **Card Gap:** 1.5rem
- **Navigation Gap:** 2rem (0.5rem on mobile)

### Effects
- **Box Shadows:**
  - Header: `0 2px 10px rgba(0, 0, 0, 0.1)`
  - Sections: `0 4px 15px rgba(0, 0, 0, 0.1)`
  - Cards: `0 3px 10px rgba(0, 0, 0, 0.2)`
  - Card Hover: `0 5px 20px rgba(0, 0, 0, 0.3)`

- **Transitions:**
  - Navigation hover: `all 0.3s ease`
  - Card transform: `transform 0.3s ease`

- **Border Radius:**
  - Sections: 10px
  - Cards: 8px
  - Navigation links: 5px

## 🔧 Key Concepts Covered

### HTML Concepts
1. **Semantic Elements:** `<header>`, `<nav>`, `<section>`, `<footer>`
2. **Document Structure:** Proper nesting and hierarchy
3. **Lists:** Ordered (`<ol>`) and unordered (`<ul>`)
4. **Links:** Anchor tags with `href` attributes
5. **IDs and Classes:** For styling and navigation

### CSS Concepts
1. **Selectors:** Class selectors (`.`), ID selectors (`#`), element selectors
2. **Box Model:** Margin, padding, border
3. **Flexbox:** For navigation layout
4. **Grid:** For responsive card layout
5. **Gradients:** Linear gradients for backgrounds
6. **Pseudo-classes:** `:hover` for interactive effects
7. **Transitions:** Smooth animations on hover
8. **Media Queries:** Responsive design for mobile

## 📚 Prerequisites

Students should have basic knowledge of:
- HTML tags and syntax
- CSS selectors and properties
- How to open files in a web browser
- Basic text editor usage

## 🎓 Learning Outcomes

After completing this activity, students will be able to:
1. Add content sections to an HTML page
2. Create navigation links between sections
3. Apply CSS classes to HTML elements
4. Style elements using various CSS properties
5. Create hover effects with CSS
6. Understand responsive design principles
7. Work with gradients and shadows
8. Organize code with proper indentation

## 🔄 Customization Options

**Easy Modifications:**
- Change color scheme by updating CSS variables
- Add more concept cards
- Modify gradient colors
- Change fonts

**Moderate Modifications:**
- Add icons to concept cards
- Create additional sections
- Implement different layout patterns
- Add more responsive breakpoints

**Advanced Modifications:**
- Add JavaScript for smooth scrolling
- Implement dark mode toggle
- Add animations on scroll
- Create interactive elements

## 💾 File Dependencies

- **index.html** → Depends on **style.css** (linked via `<link>` tag)
- No external libraries or frameworks required
- Works offline (no CDN dependencies)
- Compatible with all modern browsers

## ✅ Quality Checks

**Before Distribution:**
- [ ] All file paths are relative
- [ ] No external dependencies
- [ ] Code is properly indented
- [ ] Comments are clear and helpful
- [ ] TODOs are well-defined
- [ ] Works without live server
- [ ] Mobile responsive
- [ ] Cross-browser compatible

## 🚀 Future Enhancements

**Potential Additions:**
- Add images to make more visually engaging
- Include code snippets showing HTML/CSS/JS examples
- Add a "Try it yourself" section with live code editor
- Include video tutorial embeddings
- Add accessibility features (ARIA labels)

---

*This template is designed for W1: Front-end Design & User Experience*
*Created for Telebort Engineering educational content*
