# 🤖 AI Chat Quick Reference Card
**BudgetWise AI - Your Personal Budget Advisor**

---

## 🚀 Quick Start

```bash
# 1. Install
pip install google-generativeai python-dotenv

# 2. Get API Key
https://makersuite.google.com/app/apikey

# 3. Set Key (choose one)
echo GEMINI_API_KEY=your_key > .env
# OR enter in app sidebar

# 4. Run
streamlit run app/budgetwise_app.py
```

---

## 💬 Sample Questions

### Budget Planning
```
"What should my monthly budget be?"
"Help me create a realistic daily spending limit"
"How much should I allocate to different categories?"
```

### Spending Analysis
```
"Where am I overspending?"
"Analyze my spending patterns for the last month"
"Which day of the week do I spend the most?"
```

### Saving Strategies
```
"How can I save ₹50,000 in 6 months?"
"What's the easiest way to reduce my expenses?"
"Help me cut my food expenses by 20%"
```

### Goal Setting
```
"I want to save for a vacation. Help me plan."
"Create a 3-month emergency fund plan"
"How can I save ₹1 lakh in a year?"
```

### Category Optimization
```
"I'm spending too much on dining out. What should I do?"
"Help me optimize my transportation costs"
"Review my entertainment expenses"
```

---

## ⚡ Quick Insights

**Generate instant AI analysis:**
1. Go to "🤖 AI Chat" tab
2. Click "⚡ Quick Insights"
3. Click "🔄 Generate Fresh Insights"

**You get:**
- 📊 Spending pattern analysis
- 💡 Saving opportunities  
- 🎯 Budget recommendations

---

## 💰 Savings Planner

**Create custom savings plan:**
1. Go to "💰 Savings Plan" tab
2. Enter goal amount (₹)
3. Select time period (months)
4. Click "🚀 Create Savings Plan"

**AI generates:**
- ✅ Feasibility assessment
- ✅ Monthly targets
- ✅ Actionable strategies
- ✅ Month-by-month milestones

---

## 🎯 Tips for Best Results

### Do:
✅ Be specific in your questions
✅ Mention your goals clearly
✅ Ask follow-up questions
✅ Use actual numbers
✅ Request actionable advice

### Don't:
❌ Ask about investments (not licensed)
❌ Expect guaranteed outcomes
❌ Share sensitive personal info
❌ Ask about other people's finances

---

## 🔑 API Key Help

### Get Free Key:
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy and save securely

### Add to App:
**Method 1: .env file (recommended)**
```bash
GEMINI_API_KEY=your_api_key_here
```

**Method 2: Sidebar**
- Open app
- Look for "🔑 API Configuration" in sidebar
- Paste key

### Free Limits:
- ⚡ 15 requests/minute
- 📅 1,500 requests/day
- 💰 Completely FREE!

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Gemini AI not available" | `pip install google-generativeai` |
| "API key not provided" | Add key to `.env` or sidebar |
| "Rate limit exceeded" | Wait 1 minute, retries automatically |
| "Failed to initialize" | Check API key validity |
| No response | Check internet connection |

---

## 📊 Feature Tabs

### 💬 Chat
- Natural conversations
- Suggested questions
- Chat history
- Clear chat option

### ⚡ Quick Insights  
- One-click analysis
- Pattern detection
- Saving tips
- Budget advice

### 💰 Savings Plan
- Goal-based planning
- Timeline selection
- Feasibility check
- Action steps

---

## 🎨 Interface Guide

**Sidebar:**
- 🔑 API Configuration
- 📄 Page Navigation
- 📊 Quick Stats

**Main Area:**
- 💬 Chat messages
- 🤖 AI responses
- 📊 Insights cards
- 💰 Savings metrics

**Chat Input:**
- Type question
- Press Enter
- Or click suggested questions

---

## 💡 Example Interaction

```
👤 You: "Analyze my spending patterns"

🤖 AI: "Based on your data, I see these patterns:

1. Average daily spending: ₹2,450
2. Weekend spike: +35% (₹3,300)
3. Top category: Dining (28% of expenses)
4. Trend: Increasing 12% month-over-month

Recommendations:
• Set a ₹2,200 daily budget
• Reduce dining out by 2 meals/week
• Track weekend spending carefully
• Consider meal prep for weekdays

This could save you ~₹8,000/month!"

👤 You: "Help me create a meal prep plan"

🤖 AI: "Great idea! Here's a practical meal prep strategy..."
```

---

## 🔒 Privacy Note

✅ **Your data stays local**
✅ **Only summaries sent to AI**
✅ **No personal info shared**
✅ **Session-based chat history**
✅ **Secure API key storage**

---

## 📚 More Help

- 📖 Full Guide: `docs/AI_CHAT_GUIDE.md`
- 🔧 Setup Script: `python setup_ai_chat.py`
- 💻 Code: `src/gemini_advisor.py`
- 🐛 Issues: GitHub repository

---

**Happy Budgeting with AI! 💰✨**

*© 2025 Mohammed Arfath - BudgetWise AI*
